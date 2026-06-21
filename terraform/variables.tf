variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "The target AWS region for deployment sandbox"
}

variable "bucket_name" {
  type        = string
  default     = "mlflow-model-registry"
  description = "Unique global name of our storage bucket"
}