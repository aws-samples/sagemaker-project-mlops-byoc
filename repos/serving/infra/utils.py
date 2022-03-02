import boto3

sm_client = boto3.client("sagemaker")

def get_pipeline_execution_arn(model_package_arn: str):
    """Geturns the execution arn for the latest approved model package

    Args:
        model_package_arn: The arn of the model package

    Returns:
        The arn of the sagemaker pipeline that created the model package.
    """

    artifact_arn = sm_client.list_artifacts(SourceUri=model_package_arn)[
        "ArtifactSummaries"
    ][0]["ArtifactArn"]
    return sm_client.describe_artifact(ArtifactArn=artifact_arn)["MetadataProperties"][
        "GeneratedBy"
    ]

def get_processing_output(
    pipeline_execution_arn: str,
    step_name: str = "DataQualityBaseline",
    output_name: str = "data_baseline",
 ):
    """Filters the model packages based on a list of model package versions.

    Args:
        pipeline_execution_arn: The pipeline execution arn
        step_name: The optional processing step name
        output_name: The output value to pick from the processing job

    Returns:
        The outputs from the processing job
    """

    steps = sm_client.list_pipeline_execution_steps(
        PipelineExecutionArn=pipeline_execution_arn
    )["PipelineExecutionSteps"]

    processing_job_arn = [
        s["Metadata"]["ProcessingJob"]["Arn"]
        for s in steps
        if s["StepName"] == step_name
    ][0]

    processing_job_name = processing_job_arn.split("/")[-1]
    outputs = sm_client.describe_processing_job(ProcessingJobName=processing_job_name)[
        "ProcessingOutputConfig"
    ]["Outputs"]
    return [o["S3Output"]["S3Uri"] for o in outputs if o["OutputName"] == output_name][
        0
    ]

def get_model_package_arn(model_package_group_name: str):
    return sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    )["ModelPackageSummaryList"][0]["ModelPackageArn"]