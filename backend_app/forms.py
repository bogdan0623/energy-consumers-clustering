from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError


class CreateUserForm(UserCreationForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if len(username) < 5:  # minimum 5 characters
            raise ValidationError("Username must be at least 5 characters long")
        if User.objects.filter(username=username).exists():
            raise ValidationError("Username is already in use")
        return username

    def clean_email(self):
        email = self.cleaned_data.get('email')
        if not email:
            raise ValidationError("Email is required")
        if User.objects.filter(email=email).exists():
            raise ValidationError("Email is already in use")
        return email

    def clean_password1(self):
        password1 = self.cleaned_data.get('password1')
        if len(password1) < 8:
            raise ValidationError("Password must be at least 8 characters long")
        return password1

    def clean_password2(self):
        password2 = self.cleaned_data.get('password2')
        password1 = self.cleaned_data.get('password1')
        if password1 and password2 and password1 != password2:
            raise ValidationError("Passwords do not match")
        return password2