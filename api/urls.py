from django.urls import include, path
from .views import *

urlpatterns = [
    path('inference', InferenceCreateAPIView.as_view()),
    path('shap/<str:id>', get_explaination_shap_url),
]