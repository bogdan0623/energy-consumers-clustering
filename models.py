from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

# Create your models here.


class Account(models.Model):
    ROLES = (
        ('consumer', 'Consumer'),
        ('researcher', 'Researcher'),
    )
    username = models.CharField(max_length=15)
    password = models.CharField(max_length=20)

    class Meta:
        abstract = True


class Consumer(Account):
    pod = models.CharField(max_length=8, default="")
    role = models.CharField(choices=Account.ROLES, default="consumer", max_length=15)


class Researcher(Account):
    role = models.CharField(choices=Account.ROLES, default="researcher", max_length=15)


class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    assigned_pod = models.CharField(max_length=100, blank=True, null=True)


@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.userprofile.save()

