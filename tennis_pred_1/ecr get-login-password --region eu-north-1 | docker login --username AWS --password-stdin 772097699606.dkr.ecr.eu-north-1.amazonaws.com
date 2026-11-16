{
    "repository": {
        "repositoryArn": "arn:aws:ecr:eu-north-1:772097699606:repository/tennis-pred",
        "registryId": "772097699606",
        "repositoryName": "tennis-pred",
        "repositoryUri": "772097699606.dkr.ecr.eu-north-1.amazonaws.com/tennis-pred",
        "createdAt": "2026-11-16T17:27:42.710000+00:00",
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
