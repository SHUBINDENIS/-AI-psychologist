from django.core.validators import URLValidator
from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Questions(models.Model):
    class Meta:
        db_table = 'questions'
        verbose_name = 'Список Вопросов'
        verbose_name_plural = verbose_name

    question = models.TextField(
        blank=False,
        null=False,
        unique=True,
        verbose_name='Вопрос'
    )

    def __str__(self):
        return f'Вопрос {self.id}'


class CustomUser(models.Model):
    class Meta:
        db_table = 'users'
        verbose_name = 'Пользователь'
        verbose_name_plural = 'Пользователи'

    user_name = models.TextField(
        blank=False,
        null=False,
        unique=True,
        verbose_name='Имя пользователя'
    )
    user_email = models.EmailField(
        blank=True,
        null=True,
        unique=False,
        verbose_name='Почта'
    )
    user_password = models.TextField(
        blank=False,
        null=False,
        unique=True,
        verbose_name='Пароль'
    )

    def __str__(self):
        return self.user_name


class FormTargets(models.Model):
    vk_link = models.URLField(verbose_name="VK Profile URL")

    extroversion = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Extroversion"
    )
    agreeableness = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Agreeableness (Tendency to Consensus)"
    )
    conscientiousness = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Conscientiousness (Impulse Control)"
    )
    neuroticism = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Neuroticism (Negative Emotionality)"
    )
    openness = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Openness to Experience"
    )
    negative_affectivity = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Negative Affectivity (Anxiety, Depression)"
    )
    detachment = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Detachment (Low Emotionality, Social Isolation)"
    )
    antagonism = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Antagonism (Hostility, Manipulativeness)"
    )
    disinhibition = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Disinhibition (Impulsivity, Irresponsibility)"
    )
    psychoticism = models.FloatField(
        null=True,
        blank=True,
        verbose_name="Psychoticism (Eccentricity, Unusual Experiences)"
    )
    extraversion_introversion = models.CharField(
        max_length=1,
        null=True,
        blank=True,
        verbose_name="Extraversion-Introversion"
    )
    logic_ethics = models.CharField(
        max_length=1,
        null=True,
        blank=True,
        verbose_name="Logic-Ethics"
    )
    intuition_sensing = models.CharField(
        max_length=1,
        null=True,
        blank=True,
        verbose_name="Intuition-Sensing"
    )
    rationality_irrationality = models.CharField(
        max_length=1,
        null=True,
        blank=True,
        verbose_name="Rationality-Irrationality"
    )

    def __str__(self):
        return f"Страница ВК: {self.vk_link}"



class Link(models.Model):
    class Meta:
        db_table = 'link'
        verbose_name = 'Ссылка'
        verbose_name_plural = 'Ссылки'

    user_id = models.ForeignKey(
        CustomUser,
        on_delete=models.CASCADE,
        blank=False,
        null=False,
        verbose_name='id пользователя'
    )
    first_name = models.TextField(
        blank=False,
        null=False,
        verbose_name='имя'
    )
    last_name = models.TextField(
        blank=False,
        null=False,
        verbose_name='фамилия'
    )
    vk_link = models.TextField(
        validators=[URLValidator()],
        blank=False,
        null=False,
        verbose_name='ссылка на ВКонтакте'
    )

    def __str__(self):
        return f'{self.first_name} {self.last_name}'


class Post(models.Model):
    class Meta:
        db_table = 'post'
        verbose_name = 'Пост'
        verbose_name_plural = 'Посты'

    vk_id = models.TextField(
        blank=False,
        null=False,
        verbose_name='id пользователя в вк'
    )
    date = models.DateField(
        blank=False,
        null=False,
        verbose_name='дата публикации'
    )
    POST_TYPE_CHOICES = [
        ('post', 'Пост'),
        ('repost', 'Репост'),
    ]
    post_type = models.TextField(
        blank=False,
        null=False,
        choices=POST_TYPE_CHOICES,
        verbose_name='тип поста'
    )
    attachments = models.TextField(
        blank=True,
        null=True,
        verbose_name='приложение к посту'
    )
    text = models.TextField(
        blank=True,
        null=True,
        verbose_name='текст поста'
    )
    SENTIMENT_TYPE_CHOICES = [
        ('positive', 'Позитивная'),
        ('negative', 'Негативная'),
        ('neutral', 'Нейтральная'),
    ]
    sentiment = models.TextField(
        blank=True,
        null=True,
        verbose_name='оттенок',
        choices=SENTIMENT_TYPE_CHOICES
    )

    def __str__(self):
        return f'{self.vk_id} {self.sentiment}'


class PatientAnalysis(models.Model):
    # user = models.ForeignKey(User, on_delete=models.CASCADE, default=1, verbose_name='Пользователь')
    patient_first_name = models.CharField(max_length=100, verbose_name='Имя пациента')
    patient_last_name = models.CharField(max_length=100, verbose_name='Фамилия пациента')
    vk_link = models.URLField(verbose_name='Ссылка на VK')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='Время анализа')

    class Meta:
        verbose_name = 'Анализ пациента'
        verbose_name_plural = 'Анализы пациентов'
        ordering = ['-created_at']  # Сортировка по убыванию даты

    def __str__(self):
        return f"{self.patient_last_name} {self.patient_first_name} - {self.created_at.strftime('%d.%m.%Y %H:%M')}"
