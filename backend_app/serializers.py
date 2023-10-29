from .models import Consumer, Researcher
from rest_framework import serializers


class ConsumerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Consumer
        fields = (
            "id",
            "username",
            "password",
            "role",
            "pod"
        )


class ResearcherSerializer(serializers.ModelSerializer):
    class Meta:
        model = Researcher
        fields = (
            "id",
            "username",
            "password",
            "role"
        )

