import logging
import os

#import boto3
import sagemaker as sm
from aws_cdk import aws_apigateway as apigateway
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_lambda_python as lambda_python
from aws_cdk import aws_sagemaker as sagemaker
from aws_cdk import aws_ssm as ssm
from aws_cdk import core as cdk
from aws_cdk import aws_ecr_assets as ecr
from infra.utils import (
    get_pipeline_execution_arn,
    get_processing_output,
    get_model_package_arn
)

logger = logging.getLogger()

project_bucket_name = os.getenv("PROJECT_BUCKET")
project_name = os.getenv("SAGEMAKER_PROJECT_NAME")
project_id = os.getenv("SAGEMAKER_PROJECT_ID")
execution_role_arn = os.getenv("SAGEMAKER_PIPELINE_ROLE_ARN")
region = os.getenv("AWS_REGION")
#sm_client = boto3.client("sagemaker")
lambda_role_arn = os.getenv("LAMBDA_ROLE_ARN")


class ModelEndpointConstruct(cdk.Construct):
    def __init__(
        self,
        scope: cdk.Construct,
        construct_id: str,
        model_package_group_name: str,
        endpoint_conf: dict,
        api_gw: apigateway.RestApi,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        lambda_role = iam.Role.from_role_arn(
            self, "LambdaRole", role_arn=lambda_role_arn
        )
        
        endpoint_name = f"{project_name}-{endpoint_conf['endpoint_name']}"
        lambda_entry_point = endpoint_conf["lambda_entry_point"]
        lambda_environment = endpoint_conf["lambda_environment"]
        
        prefix = endpoint_conf["prefix"]
        schedule_config = endpoint_conf["schedule_config"]

        data_capture_sampling_percentage = schedule_config[
            "data_capture_sampling_percentage"
        ]

        data_capture_uri = f"s3://{project_bucket_name}/{prefix}/datacapture"
        async_output_uri = f"s3://{project_bucket_name}/{prefix}/async"
        
        try:
            model_package_arn = get_model_package_arn(model_package_group_name)
            variant_config_list = endpoint_conf[
                "variants"
            ]  # only one variant at the moment
            # Create variants
            variants = []
            for variant_config in variant_config_list:
                variant_name = variant_config["variant_name"]
                variant_instance_type = variant_config["instance_type"]
                variant_instance_count = variant_config["instance_count"]
                initial_variant_weight = variant_config["initial_variant_weight"]

                sagemaker_model: sagemaker.CfnModel = sagemaker.CfnModel(
                    self,
                    variant_name,
                    execution_role_arn=execution_role_arn,
                    primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                        model_package_name=model_package_arn,
                    ),
                )

                model_variant = sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    initial_instance_count=variant_instance_count,
                    initial_variant_weight=initial_variant_weight,
                    instance_type=variant_instance_type,
                    model_name=sagemaker_model.attr_model_name,
                    variant_name=variant_name,
                )

                variants.append(model_variant)
            
            model_endpoint_config = sagemaker.CfnEndpointConfig(
                self,
                f"{model_package_group_name}EndpointConfig",
                production_variants=variants,
            )

            # Enable data capture for scheduling
            model_endpoint_config.data_capture_config = sagemaker.CfnEndpointConfig.DataCaptureConfigProperty(
                enable_capture=True,
                destination_s3_uri=data_capture_uri,
                initial_sampling_percentage=data_capture_sampling_percentage,
                capture_options=[
                    sagemaker.CfnEndpointConfig.CaptureOptionProperty(
                        capture_mode="Input"
                    ),
                    sagemaker.CfnEndpointConfig.CaptureOptionProperty(
                        capture_mode="Output"
                    ),
                ],
                capture_content_type_header=sagemaker.CfnEndpointConfig.CaptureContentTypeHeaderProperty(
                    csv_content_types=["text/csv"],
                    json_content_types=["application/json"],
                ),
            )
            
            endpoint = sagemaker.CfnEndpoint(
                self,
                endpoint_name,
                endpoint_config_name=model_endpoint_config.attr_endpoint_config_name,
                endpoint_name=endpoint_name,
            )

            lambda_function = lambda_python.PythonFunction(
                self,
                f"FunctionReadOnlineFeatureStore-{endpoint_name}",
                function_name=f"sagemaker-{project_id}-EndpointFeatures",
                entry=lambda_entry_point,
                index="lambda_function.py",
                handler="lambda_handler",
                runtime=lambda_.Runtime.PYTHON_3_8,
                timeout=cdk.Duration.seconds(300),
                environment={
                    "region": region,
                    "endpoint_name": endpoint_name,
                    **lambda_environment,
                },
                role=lambda_role,
            )

            lambda_function.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "sagemaker:InvokeEndpoint",
                    ],
                    resources=[
                        f"arn:aws:sagemaker:{cdk.Aws.REGION}:{cdk.Aws.ACCOUNT_ID}:endpoint/{endpoint_name.lower()}",
                    ],
                )
            )

            lambda_function.add_to_role_policy(
                iam.PolicyStatement(
                    actions=[
                        "sagemaker:GetRecord",
                    ],
                    resources=[
                        f"*",
                    ],
                )
            )

            api_integration = apigateway.LambdaIntegration(lambda_function)

            get_endpoint = api_gw.root.add_resource(f"get-{endpoint_name}")
            get_endpoint.add_method(http_method="GET", integration=api_integration)
            endpoint_parameter = ssm.StringParameter(
                self,
                f"{endpoint_name}-URL",
                string_value=api_gw.url_for_path(path=get_endpoint.path),
                parameter_name=f"/sagemaker-{project_name}/{endpoint_name}",
            )
            endpoint_parameter.grant_read(
                iam.Role.from_role_arn(self, "SmRole", role_arn=execution_role_arn)
            )
        except:
            logger.exception("No suitable model version found")
        
        #TODO: support model monitor