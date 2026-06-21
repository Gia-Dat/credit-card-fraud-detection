output "s3_bucket_arn" {
  value       = aws_s3_bucket.mlflow_bucket.arn
  description = "The Amazon Resource Name of the created S3 storage block"
}

output "s3_bucket_name" {
  value       = aws_s3_bucket.mlflow_bucket.id
  description = "The exact name of our provisioned bucket"
}