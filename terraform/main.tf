resource "aws_s3_bucket" "mlflow_bucket" {
  bucket        = var.bucket_name
  force_destroy = true # Allows clean deletion of bucket files later if needed

  tags = {
    Environment = "Development"
    Project     = "Credit-Card-Fraud-Detection"
  }
}