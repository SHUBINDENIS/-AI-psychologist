# from main_app.views import *
# from django.urls import path
#
# urlpatterns = [
#     path('', start_page, name='start_page'),
#     path('api/v1/main/', display_main, name='main'),
#     path('api/v1/search', display_search, name='search'),
#     path('api/v1/form', display_form, name='form'),
#     path('api/v1/model', display_model, name='model'),
#     path('api/v1/login', display_profile, name='profile'),
# ]

from django.urls import path
from django.contrib import auth
from main_app.views import (
    StartPageAPI,
    MainPageAPI,
    SearchAPI,
    SurveyFormAPI,
    ModelAPI,
    ProfileAPI,
    HistoryView,
    PatientAnalysisAPI
)
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', StartPageAPI.as_view(), name='start_page'),
    path('api/v1/main/', MainPageAPI.as_view(), name='main'),
    path('api/v1/search/', SearchAPI.as_view(), name='search'),
    path('api/v1/form/', SurveyFormAPI.as_view(), name='form'),
    path('api/v1/model/', ModelAPI.as_view(), name='model'),
    path('api/v1/patient-analysis/', PatientAnalysisAPI.as_view(), name='patient-analysis'),
    
    path('api/v1/login/', ProfileAPI.as_view(), name='profile'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', auth_views.LogoutView.as_view(next_page='home'), name='logout'),
    path('profile/', ProfileAPI.as_view(), name='profile'),
    path('api/register/', views.RegisterView.as_view(), name='api-register'),
    path('api/login/', views.LoginView.as_view(), name='api-login'),
    path('api/logout/', views.LogoutView.as_view(), name='api-logout'),
    path('api/profile/', views.ProfileView.as_view(), name='api-profile'),
    path('api/history/', views.HistoryView.as_view(), name='api-history'),
]
