from django.shortcuts import redirect, render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from main_app.models import Questions, PatientAnalysis, FormTargets
from main_app.helpers import parsing_plus_prediction, convert_answers_in_targets
from django.views.generic import TemplateView
import numpy as np
import pandas as pd
from django.utils import timezone
from datetime import timedelta
from django.contrib import messages
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .forms import UserRegisterForm, UserLoginForm
from rest_framework import generics
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import User
from .serializers import RegisterSerializer, ProfileSerializer, UserSerializer


# Страница перенаправления (можно использовать для редиректа или выдачи ссылки)
class StartPageAPI(APIView):
    def get(self, request):
        # Возвращаем URL главной страницы API
        return Response(
            {"redirect_url": "/api/main/"},
            status=status.HTTP_200_OK
        )


# Обработчик отсутствующей страницы (404)
def page_not_found_api(request, exception):
    return Response(
        {"error": "Ошибка. Страница не существует."},
        status=status.HTTP_404_NOT_FOUND
    )


# Главная страница API
class MainPageAPI(APIView):
    def get(self, request):
        return Response(
            {"message": "Это главная страница приложения."},
            status=status.HTTP_200_OK
        )


# Эндпоинт для поиска
class SearchAPI(APIView):
    # permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        return Response(
            {"instruction": "Отправьте POST-запрос с полями 'patient-firstname', 'patient-lastname' и 'patient-vk'."},
            status=status.HTTP_200_OK
        )

    def post(self, request):
        firstname = request.data.get('patient-firstname', '')
        lastname = request.data.get('patient-lastname', '')
        vk_link = request.data.get('patient-vk', '')

        # try:
        # Сохраняем информацию об анализе
        # Используем request.user.id вместо request.user

        # Выполняем обработку запроса и предсказание
        model_result = parsing_plus_prediction(vk_link)

        PatientAnalysis.objects.create(
            patient_first_name=firstname,
            patient_last_name=lastname,
            vk_link=vk_link
        )

        result_json = {
            "Экстраверсия": model_result[0],
            "Негативная аффективность (тревожность, депрессия)": model_result[1],
            "Экстраверсия – интроверсия": model_result[2],
            "Доброжелательность (Склонность к согласию)": model_result[3],
            "Отстраненность (низкая эмоциональность, социальная изоляция)": model_result[4],
            "Логика – этика": model_result[5],
            "Добросовестность (Контроль импульсивности)": model_result[6],
            "Антагонизм (враждебность, манипулятивность)": model_result[7],
            "Интуиция – сенсорика": model_result[8],
            "Негативная эмоциональность (Нейротизм)": model_result[9],
            "Дезингуляция (импульсивность, безответственность)": model_result[10],
            "Рациональность – иррациональность": model_result[11],
            "Открытость опыту": model_result[12],
            "Психотизм (эксцентричность, необычные переживания)": model_result[13],
            "Тип личности": model_result[14]
        }

        return Response(result_json, status=status.HTTP_200_OK)

        # except Exception as e:
        #     print('Произошла ошибка:', str(e))
        #     return Response(
        #         {"error_message": "На сервере произошла внутренняя ошибка. Пожалуйста, попробуйте еще раз позже."},
        #         status=status.HTTP_500_INTERNAL_SERVER_ERROR
        #     )


# Эндпоинт для опроса (форма)
class SurveyFormAPI(APIView):
    def get(self, request):
        # Получаем все вопросы из базы данных
        questions = Questions.objects.all().order_by('id')
        # Формируем данные с разделением ID и вопросов
        data = {
            "question": [{"id": question.id,
                          "patient-vk" if question.id == 0
                          else "question": question.question}
                         for question in questions]
        }
        return Response(
            data,
            status=status.HTTP_200_OK
        )

    def post(self, request):
        # Передаём все данные формы в функцию конвертации
        vk_link, targets_dict = convert_answers_in_targets(request.data)

        # Проверяем, что функция вернула vk_link и нужное количество таргетов
        if not vk_link or not targets_dict or len(targets_dict) != 14:
            return Response(
                {"error": "Некорректные данные формы"},
                status=status.HTTP_400_BAD_REQUEST
            )
# Проверяем, есть ли уже запись с таким vk_link
        form_targets, created = FormTargets.objects.update_or_create(
            vk_link=vk_link,
            defaults={
                'extroversion': targets_dict['extroversion'],
                'agreeableness': targets_dict['agreeableness'],
                'conscientiousness': targets_dict['conscientiousness'],
                'neuroticism': targets_dict['neuroticism'],
                'openness': targets_dict['openness'],
                'negative_affectivity': targets_dict['negative_affectivity'],
                'detachment': targets_dict['detachment'],
                'antagonism': targets_dict['antagonism'],
                'disinhibition': targets_dict['disinhibition'],
                'psychoticism': targets_dict['psychoticism'],
                'extraversion_introversion': targets_dict['extraversion_introversion'],
                'logic_ethics': targets_dict['logic_ethics'],
                'intuition_sensing': targets_dict['intuition_sensing'],
                'rationality_irrationality': targets_dict['rationality_irrationality']
            }
        )

# created будет True, если запись только что создана, и False если обновлена существующая



        return Response(
            {
                "vk_link": vk_link,
                "targets": targets_dict,
                "status": "Saved successfully"
            },
            status=status.HTTP_201_CREATED
        )


# Эндпоинт для модели (пример вывода модели)
class ModelAPI(APIView):
    def get(self, request):
        # Например, можно вернуть информацию о модели или ссылки на ресурсы
        return Response(
            {"message": "Информация о модели доступна здесь."},
            status=status.HTTP_200_OK
        )


# Эндпоинт для профиля пользователя
class ProfileAPI(APIView):
    # Если требуется проверка авторизации, добавьте permission_classes
    # permission_classes = [permissions.IsAuthenticated]
    def get(self, request):
        return Response(
            {"message": "Это тестовая страница профиля."},
            status=status.HTTP_200_OK
        )


class PageNotFoundView(TemplateView):
    template_name = '404.html'

    def render_to_response(self, context, **response_kwargs):
        response_kwargs['status'] = 404
        return super().render_to_response(context, **response_kwargs)


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Регистрация успешна!')
            return redirect('home')
    else:
        form = UserRegisterForm()
    return render(request, 'main_app/register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, 'Вход выполнен успешно!')
                return redirect('home')
            else:
                messages.error(request, 'Неверное имя пользователя или пароль')
    else:
        form = UserLoginForm()
    return render(request, 'main_app/login.html', {'form': form})

@login_required
def profile(request):
    return render(request, 'main_app/profile.html')

class RegisterView(generics.CreateAPIView):
    queryset = User.objects.all()
    permission_classes = (permissions.AllowAny,)
    serializer_class = RegisterSerializer

class LoginView(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data,
                                         context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'token': token.key,
            'user_id': user.pk,
            'username': user.username,
            'email': user.email
        })

class ProfileView(generics.RetrieveUpdateAPIView):
    serializer_class = ProfileSerializer
    permission_classes = (permissions.IsAuthenticated,)

    def get_object(self):
        return self.request.user

    def get(self, request, *args, **kwargs):
        user = self.get_object()
        serializer = self.get_serializer(user)
        return Response(serializer.data)

    def patch(self, request, *args, **kwargs):
        return self.partial_update(request, *args, **kwargs)

class HistoryView(generics.ListAPIView):
    # permission_classes = (permissions.IsAuthenticated,)

    def get(self, request):
        # Получаем все анализы текущего пользователя
        analyses = PatientAnalysis.objects.filter(user=request.user)
        history = [{
            'patient_name': f"{analysis.patient_first_name} {analysis.patient_last_name}",
            'vk_link': analysis.vk_link,
            'last_analysis': (analysis.created_at + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M')
        } for analysis in analyses]
        return Response(history)

class LogoutView(generics.GenericAPIView):
    permission_classes = (permissions.IsAuthenticated,)

    def post(self, request):
        try:
            request.user.auth_token.delete()
            return Response({"message": "Успешный выход из системы"}, 
                          status=status.HTTP_200_OK)
        except Exception:
            return Response({"message": "Ошибка при выходе из системы"}, 
                          status=status.HTTP_400_BAD_REQUEST)


class PatientAnalysisAPI(APIView):
    # permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        # Получаем последние 5 записей, отсортированных по убыванию времени
        analyses = PatientAnalysis.objects.all().order_by('-created_at')[:5]
        
        # Формируем данные для ответа
        data = []
        for analysis in analyses:
            data.append({
                'id': analysis.id,
                'patient_first_name': analysis.patient_first_name,
                'patient_last_name': analysis.patient_last_name,
                'vk_link': analysis.vk_link,
                'created_at': analysis.created_at.strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Добавляем сообщение "записи нет" для каждой недостающей записи
        for _ in range(5 - len(analyses)):
            data.append({"message": "записи нет"})
        
        return Response(data, status=status.HTTP_200_OK)

    def post(self, request):
        vk_link = request.data.get('patient-vk', '')
        
        if not vk_link:
            return Response({"error": "Не указана ссылка на VK"}, status=status.HTTP_400_BAD_REQUEST)
        
        # Получаем все записи для указанной ссылки на VK
        analyses = PatientAnalysis.objects.filter(vk_link=vk_link).order_by('-created_at')[:5]
        
        # Формируем данные для ответа
        data = []
        for analysis in analyses:
            data.append({
                'id': analysis.id,
                'patient_first_name': analysis.patient_first_name,
                'patient_last_name': analysis.patient_last_name,
                'vk_link': analysis.vk_link,
                'created_at': timezone.localtime(analysis.created_at).strftime('%Y-%m-%d %H:%M')
            })
        
        # Добавляем сообщение "записи нет" для каждой недостающей записи
        for _ in range(5 - len(analyses)):
            data.append({"message": "записи нет"})
        
        return Response(data, status=status.HTTP_200_OK) 