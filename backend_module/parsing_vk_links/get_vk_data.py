import requests
import pandas as pd
from datetime import datetime
from django.conf import settings
from typing import Union, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)

VK_TOKEN = settings.VK_TOKEN
API_VERSION = '5.199'
WALL_STEP = 100


def id_from_uri(uri: str) -> str:
    if uri.startswith('https://vk.com/'):
        return uri[15:].strip()
    logging.warning(f'Invalid VK link: {uri}')
    return ''


def read_vk(user_id: str) -> Union[requests.Response, None]:
    try:
        url = 'https://api.vk.com/method/users.get'
        params = {
            'user_ids': user_id,
            'fields': ','.join([
                'photo_max_orig', 'sex', 'first_name', 'last_name', 'bdate',
                'status', 'city', 'country', 'contacts', 'has_photo', 'has_mobile',
                'home_town', 'can_post', 'can_see_all_posts', 'can_see_audio',
                'interests', 'books', 'tv', 'quotes', 'games', 'movies', 'activities',
                'music', 'can_write_private_message', 'can_send_friend_request',
                'can_be_invited_group', 'site', 'counters', 'screen_name',
                'education', 'occupation', 'personal', 'relation'
            ]),
            'access_token': VK_TOKEN,
            'v': API_VERSION
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response
    except Exception as e:
        logging.error(f"Ошибка при получении данных пользователя {user_id}: {e}")
        return None


def read_wall(user_id: str) -> List[dict]:
    posts = []
    offset = 0
    total_count = 1
    session = requests.Session()

    while offset < total_count:
        try:
            url = 'https://api.vk.com/method/wall.get'
            params = {
                'owner_id': user_id,
                'offset': offset,
                'count': WALL_STEP,
                'filter': 'owner',
                'access_token': VK_TOKEN,
                'v': API_VERSION
            }
            response = session.get(url, params=params)
            response.raise_for_status()
            data = response.json().get('response', {})
            total_count = data.get('count', 0)

            for post in data.get('items', []):
                attachments = " ".join(a.get('type', '') for a in post.get('attachments', []))
                post_type = 'repost' if 'copy_history' in post else 'post'
                post_date = datetime.fromtimestamp(post.get('date')).strftime("%d.%m.%Y %H:%M:%S")
                posts.append({
                    'user_id': user_id,
                    'date': post_date,
                    'type': post_type,
                    'attachments': attachments,
                    'text': post.get('text', '')
                })
        except Exception as e:
            logging.error(f"Ошибка при получении постов для {user_id}: {e}")
            break
        offset += WALL_STEP

    return posts


def json2df(user_response: requests.Response) -> pd.DataFrame:
    try:
        data = user_response.json().get('response', [])
        users_data = [{
            'user_id': user.get('id'),
            'photo_max_orig': user.get('photo_max_orig'),
            'sex': user.get('sex'),
            'first_name': user.get('first_name'),
            'last_name': user.get('last_name'),
            'bdate': user.get('bdate', ''),
            'relation': user.get('relation'),
            'country': user.get('country', {}).get('title', ''),
            'city': user.get('city', {}).get('title', ''),
            'mobile_phone': user.get('contacts', {}).get('mobile_phone', ''),
            'home_phone': user.get('contacts', {}).get('home_phone', ''),
            'has_photo': user.get('has_photo'),
            'has_mobile': user.get('has_mobile'),
            'home_town': user.get('home_town'),
            'can_post': user.get('can_post'),
            'can_see_all_posts': user.get('can_see_all_posts'),
            'can_see_audio': user.get('can_see_audio'),
            'interests': user.get('interests'),
            'books': user.get('books'),
            'tv': user.get('tv'),
            'quotes': user.get('quotes'),
            'games': user.get('games'),
            'movies': user.get('movies'),
            'activities': user.get('activities'),
            'music': user.get('music'),
            'can_write_private_message': user.get('can_write_private_message'),
            'can_send_friend_request': user.get('can_send_friend_request'),
            'can_be_invited_group': user.get('can_be_invited_group'),
            'site': user.get('site'),
            'status': user.get('status'),
            'albums': user.get('counters', {}).get('albums', ''),
            'videos': user.get('counters', {}).get('videos', ''),
            'audios': user.get('counters', {}).get('audios', ''),
            'photos': user.get('counters', {}).get('photos', ''),
            'notes': user.get('counters', {}).get('notes', ''),
            'gifts': user.get('counters', {}).get('gifts', ''),
            'friends': user.get('counters', {}).get('friends', ''),
            'groups': user.get('counters', {}).get('groups', ''),
            'followers': user.get('counters', {}).get('followers', ''),
            'pages': user.get('counters', {}).get('pages', ''),
            'subscriptions': user.get('counters', {}).get('subscriptions', ''),
            'screen_name': user.get('screen_name'),
            'university': user.get('education', {}).get('university_name', ''),
            'occupation_type': user.get('occupation', {}).get('type', ''),
            'occupation_name': user.get('occupation', {}).get('name', ''),
            'political': user.get('personal', {}).get('political', ''),
            'religion': user.get('personal', {}).get('religion', ''),
            'inspired_by': user.get('personal', {}).get('inspired_by', ''),
            'people_main': user.get('personal', {}).get('people_main', ''),
            'life_main': user.get('personal', {}).get('life_main', ''),
            'smoking': user.get('personal', {}).get('smoking', ''),
            'alcohol': user.get('personal', {}).get('alcohol', ''),
            'can_access_closed': user.get('can_access_closed'),
            'is_closed': user.get('is_closed')
        } for user in data]
        return pd.DataFrame(users_data)
    except Exception as e:
        logging.error(f"Ошибка при преобразовании JSON в DataFrame: {e}")
        return pd.DataFrame()


def parse_vk_data(links: Union[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(links, str):
        links = [links]

    main_data = []
    wall_data = []

    for link in links:
        user_id = id_from_uri(link)
        if not user_id:
            continue

        response = read_vk(user_id)
        if response and response.status_code == 200:
            user_df = json2df(response)
            if user_df.empty:
                continue
            if pd.isna(user_df.loc[0, 'screen_name']):
                user_df.loc[0, 'screen_name'] = link.split('/')[-1]
            main_data.append(user_df)
        else:
            logging.warning(f"Не удалось получить данные пользователя по ссылке: {link}")

        posts = read_wall(user_id)
        if posts:
            wall_data.extend(posts)

    if main_data:
        return pd.concat(main_data, ignore_index=True), pd.DataFrame(wall_data)
    return pd.DataFrame(), pd.DataFrame(wall_data)
