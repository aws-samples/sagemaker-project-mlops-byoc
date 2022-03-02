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
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subscriptions

from aws_cdk import aws_applicationautoscaling as app_autoscaling

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
lambda_role_arn = os.getenv("LAMBDA_ROLE_ARN")

class ModelAsyncEndpointConstruct(cdk.Construct):
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
        
        sagemaker_role = iam.Role.from_role_arn(
            self, "SageMakerRole", role_arn=execution_role_arn
        )
        
        endpoint_name = f"{project_name}-{endpoint_conf['endpoint_name']}-async"
        lambda_entry_point = endpoint_conf["lambda_entry_point"]
        lambda_environment = endpoint_conf["lambda_environment"]
        
        prefix = endpoint_conf["prefix"]
        app_autoscaling_config = endpoint_conf["app_autoscaling"]
        
        min_capacity = app_autoscaling_config["min_capacity"]
        max_capacity = app_autoscaling_config["max_capacity"]
        scale_in_cooldown = app_autoscaling_config["scale_in_cooldown"]
        scale_out_cooldown = app_autoscaling_config["scale_out_cooldown"]
        target_value = app_autoscaling_config["target_value"]
        metric_name = app_autoscaling_config["metric_name"]
        statistic = app_autoscaling_config["statistic"]

        data_capture_uri = f"s3://{project_bucket_name}/{prefix}/async/datacapture"
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
            
            # Create topics
            success_topic = sns.Topic(self, f"{project_name}-topic-success",
                display_name="Asynchronous inference succeeded"
            )
            
            error_topic = sns.Topic(self, f"{project_name}-topic-error",
                display_name="Asynchronous inference Failed"
            )
            
            for email_addr in endpoint_conf["subscribers"]:
                success_topic.add_subscription(subscriptions.EmailSubscription(email_addr))
                error_topic.add_subscription(subscriptions.EmailSubscription(email_addr))
            
            async_inference_config=sagemaker.CfnEndpointConfig.AsyncInferenceConfigProperty(
                output_config=sagemaker.CfnEndpointConfig.AsyncInferenceOutputConfigProperty(
                    s3_output_path=async_output_uri,
                    # the properties below are optional
                    notification_config=sagemaker.CfnEndpointConfig.AsyncInferenceNotificationConfigProperty(
                        success_topic=success_topic.topic_arn,
                        error_topic=error_topic.topic_arn
                    )
                ),

                # the properties below are optional
                client_config=sagemaker.CfnEndpointConfig.AsyncInferenceClientConfigProperty(
                    max_concurrent_invocations_per_instance=endpoint_conf["max_concurrent_invocations_per_instance"]
                )
            )
            
            model_endpoint_config = sagemaker.CfnEndpointConfig(
                self,
                f"{model_package_group_name}EndpointConfigAsync",
                production_variants=variants,
                async_inference_config=async_inference_config
            )

            endpoint = sagemaker.CfnEndpoint(
                self,
                endpoint_name,
                endpoint_config_name=model_endpoint_config.attr_endpoint_config_name,
                endpoint_name=endpoint_name,
            )
            
            success_topic.grant_publish(sagemaker_role)
            error_topic.grant_publish(sagemaker_role)
            
            """ the error of incorrect variant name occurred when triggering autoscaling
            
            resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
            
            target = app_autoscaling.ScalableTarget(self, "ScalableTarget",
                service_namespace=app_autoscaling.ServiceNamespace.SAGEMAKER,
                max_capacity=max_capacity,
                min_capacity=min_capacity,
                resource_id=resource_id,
                scalable_dimension="sagemaker:variant:DesiredInstanceCount"
            )
            
            dep_endpoint = cdk.ConcreteDependable()
            dep_endpoint.add(endpoint)
            target.node.add_dependency(dep_endpoint)
            
            target_configuration = app_autoscaling.CfnScalingPolicy.TargetTrackingScalingPolicyConfigurationProperty(
                target_value=target_value,
                # the properties below are optional
                customized_metric_specification=app_autoscaling.CfnScalingPolicy.CustomizedMetricSpecificationProperty(
                    metric_name=metric_name,
                    namespace="sagemaker",
                    statistic=statistic,
                    dimensions=[app_autoscaling.CfnScalingPolicy.MetricDimensionProperty(
                        name="EndpointName",
                        value=endpoint_name
                    )]
                ),
                disable_scale_in=False,
                scale_in_cooldown=scale_in_cooldown,
                scale_out_cooldown=scale_out_cooldown
            )
            
            cfn_scaling_policy = app_autoscaling.CfnScalingPolicy(self, "MyCfnScalingPolicy",
                policy_name=f"async_endpoint-{endpoint_name}",
                policy_type="TargetTrackingScaling",
                scaling_target_id=target.scalable_target_id,
                target_tracking_scaling_policy_configuration=target_configuration
            )
            """
        except:
            logger.exception("No suitable model version found")