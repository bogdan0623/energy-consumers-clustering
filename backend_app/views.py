import datetime
import os
from urllib.parse import urljoin

import matplotlib
from django.conf import settings
from django.shortcuts import render, redirect
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
from pdfkit import pdfkit
from rest_framework import status
from rest_framework.parsers import JSONParser
from django.http import JsonResponse, HttpResponse
from rest_framework.response import Response
from rest_framework.views import APIView
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, OPTICS
from sklearn.metrics import pairwise_distances, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

from .forms import CreateUserForm

matplotlib.use('Agg')
import pandas as pd

from licenta_backend.settings import BASE_DIR
from .models import Consumer, Researcher, Account
from .serializers import ConsumerSerializer, ResearcherSerializer

# Create your views here.

csv_path = r'D:\Licenta\Set date\set_date.csv'
per_day_path = r'D:\Licenta\Set date\per_day.csv'
per_month_path = r'D:\Licenta\Set date\per_month.csv'
csv_dir = r'D:\Licenta\Set date'

case12 = {}
line = {}
consume = {}
compare = {}


@csrf_exempt
def consumerApi(request, id=0):
    if request.method == 'GET':
        if id == 0:
            consumers = Consumer.objects.all()
            consumers_serializer = ConsumerSerializer(consumers, many=True)
            return JsonResponse(consumers_serializer.data, safe=False)
        else:
            consumer = Consumer.objects.get(id=id)
            consumer_serializer = ConsumerSerializer(consumer)
            return JsonResponse(consumer_serializer.data, safe=False)
    elif request.method == 'POST':
        consumer_data = JSONParser().parse(request)
        consumer_serializer = ConsumerSerializer(data=consumer_data)
        if consumer_serializer.is_valid():
            consumer_serializer.save()
            return JsonResponse("Added successfully!", safe=False)
        return JsonResponse("Failed to add!", safe=False)
    elif request.method == 'PUT':
        consumer_data = JSONParser().parse(request)
        consumer = Consumer.objects.get(id=consumer_data['id'])
        consumer_serializer = ConsumerSerializer(consumer, data=consumer_data, partial=True)
        if consumer_serializer.is_valid():
            consumer_serializer.save()
            return JsonResponse("Updated successfully!", safe=False)
        return JsonResponse("Failed to update", safe=False)
    elif request.method == 'DELETE':
        consumer = Consumer.objects.get(id=id)
        consumer.delete()
        return JsonResponse("Deleted successfully!", safe=False)


@csrf_exempt
def researcherApi(request, id=0):
    if request.method == 'GET':
        if id == 0:
            researchers = Researcher.objects.all()
            researchers_serializer = ResearcherSerializer(researchers, many=True)
            return JsonResponse(researchers_serializer.data, safe=False)
        else:
            researcher = Researcher.objects.get(id=id)
            researcher_serializer = ResearcherSerializer(researcher)
            return JsonResponse(researcher_serializer.data, safe=False)
    elif request.method == 'POST':
        researcher_data = JSONParser().parse(request)
        researcher_serializer = ResearcherSerializer(data=researcher_data)
        if researcher_serializer.is_valid():
            researcher_serializer.save()
            return JsonResponse("Added successfully!", safe=False)
        return JsonResponse("Failed to add!", safe=False)
    elif request.method == 'PUT':
        researcher_data = JSONParser().parse(request)
        researcher = Researcher.objects.get(id=researcher_data['id'])
        researcher_serializer = ResearcherSerializer(researcher, data=researcher_data, partial=True)
        if researcher_serializer.is_valid():
            researcher_serializer.save()
            return JsonResponse("Updated successfully!", safe=False)
        return JsonResponse("Failed to update", safe=False)
    elif request.method == 'DELETE':
        researcher = Researcher.objects.get(id=id)
        researcher.delete()
        return JsonResponse("Deleted successfully!", safe=False)


def convert_date_to_dhh(date):
    if date.month < 10:
        month = '0' + str(date.month)
    else:
        month = str(date.month)
    if date.day < 10:
        day = '0' + str(date.day)
    else:
        day = str(date.day)
    date_string = str(date.year) + month + day
    date_num = int(date_string) * 100 + 1

    return date_num


def filter_data(data, print_POD_list=False, pod="", d1=None, m1=None, y1=None, d2=None, m2=None, y2=None):
    if print_POD_list:
        pod_list = [*set(data['POD'].tolist())]
        pod_list.sort()
        with open("./results/PODs list.txt", "w") as f:
            for pod in pod_list:
                print(pod, file=f)
            print(len(pod_list), file=f)
        return pod_list
    if pod != "":
        selected_pod = data[data['POD'] == pod]
    else:
        selected_pod = data
    if y1 is None:
        return selected_pod
    if y2 is None:
        selected_pod_date = selected_pod[selected_pod.Year == y1]
        if m1 is not None:
            selected_pod_date = selected_pod_date[selected_pod.Month == m1]
            if d1 is not None:
                selected_pod_date = selected_pod_date[selected_pod.Day == d1]
        return selected_pod_date
    else:
        if d1 is None and d2 is None:
            selected_pod_date = selected_pod[selected_pod.Year <= y2]
            selected_pod_date = selected_pod_date[selected_pod_date.Year >= y1]
            if y1 != y2:
                selected_pod_date_sup = selected_pod_date[
                    (selected_pod_date.Month <= m2) & (selected_pod_date.Year == y2)]
                selected_pod_date_inf = selected_pod_date[
                    (selected_pod_date.Month >= m1) & (selected_pod_date.Year == y1)]
                selected_pod_date_mid = selected_pod_date[(selected_pod_date.Year > y1) & (selected_pod_date.Year < y2)]
                return pd.concat([selected_pod_date_inf, selected_pod_date_mid, selected_pod_date_sup])
            selected_pod_date = selected_pod_date[selected_pod_date.Month <= m2]
            selected_pod_date = selected_pod_date[selected_pod_date.Month >= m1]
            return selected_pod_date
        date2 = datetime.datetime(y2, m2, d2)
        date1 = datetime.datetime(y1, m1, d1)
        date_num1 = convert_date_to_dhh(date1)
        date_num2 = convert_date_to_dhh(date2)

        selected_pod_date = selected_pod[selected_pod.DHH < date_num2 + 24]
        selected_pod_date = selected_pod_date[selected_pod_date.DHH >= date_num1]
        return selected_pod_date


def month_to_string(month):
    values = {1: "JAN", 2: "FEB", 3: "MAR", 4: "APR", 5: "MAY", 6: "JUN", 7: "JUL", 8: "AUG", 9: "SEP", 10: "OCT",
              11: "NOV", 12: "DEC"}
    return values[month]


def k_means(data, n_clusters, date1, date2=None, pod="", euclidean=False, manhattan=False, cosine=False):
    X = data.loc[:, ~data.columns.isin(['Total Consume', 'POD', 'Day'])]
    if euclidean:
        dist = pairwise_distances(X, metric='euclidean')
        dist_name = 'Euclidean'
    if manhattan:
        dist = pairwise_distances(X, metric='manhattan')
        dist_name = 'Manhattan'
    if cosine:
        dist = pairwise_distances(X, metric='cosine')
        dist_name = 'Cosine'

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto').fit(dist)
    cluster_labels = kmeans.fit_predict(scaled_data)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)

    data['cluster'] = cluster_labels

    # Evaluation
    silhouette_score_average = silhouette_score(scaled_data, cluster_labels)
    davies_bouldin_index = davies_bouldin_score(scaled_data, cluster_labels)
    calinski_harabasz_sc = calinski_harabasz_score(scaled_data, cluster_labels)

    clustered_data = data

    fig, ax = plt.subplots()

    if pod == "":
        clustered_data_file = f'All_from_{str(date1.month)}_{str(date1.year)}_Kmeans_{n_clusters}_clusters_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)

        plot_name = f'Kmeans_{n_clusters}_clusters_{dist_name}.png'
        plot_path = f"./media/{plot_name}"

        # Plot for all pods
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("PODS")
        plt.title(f"K-Means {n_clusters} clusters {dist_name} {month_to_string(date1.month)}-{date1.year}")
        plt.savefig(plot_path)
    else:
        clustered_data_file = f'{pod}_from_{str(date1.year)}_Kmeans_{n_clusters}_clusters_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)

        plot_name = f'Kmeans_{pod}_{n_clusters}_clusters_{dist_name}.png'
        plot_path = f"./media/{plot_name}"

        # Plot for ONE pod
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("Day")
        plt.title(f"K-Means {n_clusters} clusters {dist_name} {date1.year} {pod}")
        plt.savefig(plot_path)
    return data, silhouette_score_average, davies_bouldin_index, calinski_harabasz_sc, clustered_data_file, plot_name


def ahc(data, n_clusters, date1, date2=None, pod="", euclidean=False, manhattan=False, cosine=False):
    X = data.loc[:, ~data.columns.isin(['Total Consume', 'POD', 'Day'])]
    if euclidean:
        dist = pairwise_distances(X, metric='euclidean')
        dist_name = 'Euclidean'
    if manhattan:
        dist = pairwise_distances(X, metric='manhattan')
        dist_name = 'Manhattan'
    if cosine:
        dist = pairwise_distances(X, metric='cosine')
        dist_name = 'Cosine'

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    ahc = AgglomerativeClustering(n_clusters=n_clusters).fit(dist)
    cluster_labels = ahc.fit_predict(scaled_data)

    data['cluster'] = cluster_labels

    # Evaluation
    silhouette_score_average = silhouette_score(scaled_data, cluster_labels)
    davies_bouldin_index = davies_bouldin_score(scaled_data, cluster_labels)
    calinski_harabasz_sc = calinski_harabasz_score(scaled_data, cluster_labels)

    clustered_data = data

    fig, ax = plt.subplots()

    if pod == "":
        clustered_data_file = f'All_from_{str(date1.month)}_{str(date1.year)}_AHC_{n_clusters}_clusters_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("PODS")
        plt.title(f"AHC {n_clusters} clusters {dist_name} {month_to_string(date1.month)}-{date1.year}")
        plot_name = f'AHC_{n_clusters}_clusters_{dist_name}.png'
        plot_path = f"./media/{plot_name}"
        plt.savefig(plot_path)
    else:
        clustered_data_file = f'{pod}_from_{str(date1.year)}_AHC_{n_clusters}_clusters_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("Day")
        plt.title(f"AHC {n_clusters} clusters {dist_name} {date1.year} {pod}")
        plot_name = f'AHC_{pod}_{n_clusters}_clusters_{dist_name}.png'
        plot_path = f"./media/{plot_name}"
        plt.savefig(plot_path)

    return data, silhouette_score_average, davies_bouldin_index, calinski_harabasz_sc, clustered_data_file, plot_name


def dbscan(data, eps, min_samples, date1, date2=None, pod="", euclidean=False, manhattan=False, cosine=False):
    X = data.loc[:, ~data.columns.isin(['Total Consume', 'POD', 'Day'])]
    if euclidean:
        dist = pairwise_distances(X, metric='euclidean')
        dist_name = 'Euclidean'
    if manhattan:
        dist = pairwise_distances(X, metric='manhattan')
        dist_name = 'Manhattan'
    if cosine:
        dist = pairwise_distances(X, metric='cosine')
        dist_name = 'Cosine'

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(dist)
    cluster_labels = dbscan.fit_predict(scaled_data)

    data['cluster'] = cluster_labels

    # Evaluation
    if len(list(set(cluster_labels))) > 1:
        silhouette_score_average = silhouette_score(scaled_data, cluster_labels)
        davies_bouldin_index = davies_bouldin_score(scaled_data, cluster_labels)
        calinski_harabasz_sc = calinski_harabasz_score(scaled_data, cluster_labels)
    else:
        silhouette_score_average = -1
        davies_bouldin_index = -1
        calinski_harabasz_sc = -1

    clustered_data = data

    fig, ax = plt.subplots()

    if pod == "":
        clustered_data_file = f'All_from_{str(date1.month)}_{str(date1.year)}_DBSCAN_min_samples_{min_samples}_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("PODS")
        plt.title(f"DBSCAN {dist_name} {month_to_string(date1.month)}-{date1.year}")
        plot_name = f'DBSCAN_min_samples_{min_samples}_{dist_name}.png'
        plot_path = f"./media/{plot_name}"
        plt.savefig(plot_path)
    else:
        clustered_data_file = f'{pod}_from_{str(date1.year)}_DBSCAN_min_samples_{min_samples}_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("Day")
        plt.title(f"DBSCAN with {dist_name} distance for {pod} {date1.year}")
        plot_name = f'DBSCAN_{pod}_min_samples_{min_samples}_{dist_name}.png'
        plot_path = f"./media/{plot_name}"
        plt.savefig(plot_path)

    return data, silhouette_score_average, davies_bouldin_index, calinski_harabasz_sc, clustered_data_file, plot_name


def optics(data, eps, min_samples, date1, date2=None, pod="", euclidean=False, manhattan=False, cosine=False):
    X = data.loc[:, ~data.columns.isin(['Total Consume', 'POD', 'Day'])]
    if euclidean:
        dist = pairwise_distances(X, metric='euclidean')
        dist_name = 'Euclidean'
    if manhattan:
        dist = pairwise_distances(X, metric='manhattan')
        dist_name = 'Manhattan'
    if cosine:
        dist = pairwise_distances(X, metric='cosine')
        dist_name = 'Cosine'

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X)

    optics = OPTICS(eps=eps, min_samples=min_samples).fit(dist)
    cluster_labels = optics.fit_predict(scaled_data)

    data['cluster'] = cluster_labels

    # Evaluation
    if len(list(set(cluster_labels))) > 1:
        silhouette_score_average = silhouette_score(scaled_data, cluster_labels)
        davies_bouldin_index = davies_bouldin_score(scaled_data, cluster_labels)
        calinski_harabasz_sc = calinski_harabasz_score(scaled_data, cluster_labels)
    else:
        silhouette_score_average = -1
        davies_bouldin_index = -1
        calinski_harabasz_sc = -1

    clustered_data = data

    fig, ax = plt.subplots()

    if pod == "":
        clustered_data_file = f'All_from_{str(date1.month)}_{str(date1.year)}_OPTICS_min_samples_{min_samples}_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("PODS")
        plt.title(f"OPTICS {dist_name} {month_to_string(date1.month)}-{date1.year}")
        plot_name = f'OPTICS_min_samples_{min_samples}_{dist_name}.png'
        plot_path = f"./media/{plot_name}"
        plt.savefig(plot_path)
    else:
        clustered_data_file = f'{pod}_from_{str(date1.year)}_OPTICS_min_samples_{min_samples}_{dist_name}.csv'
        clustered_data_path = f'./results/{clustered_data_file}'
        clustered_data.to_csv(clustered_data_path, index=False)
        sct = ax.scatter(data['Total Consume'].apply(pd.to_numeric), data.index, c=data['cluster'], cmap='viridis')
        legend1 = ax.legend(*sct.legend_elements(), loc="best", title="Clusters")
        ax.add_artist(legend1)
        plt.xlabel("Energy consumption(kWh)")
        plt.ylabel("Day")
        plt.title(f"OPTICS {dist_name} {date1.year}")
        plot_name = f'OPTICS_{pod}_min_samples_{min_samples}_{dist_name}.png'
        plot_path = f"./media/{plot_name}"
        plt.savefig(plot_path)

    return data, silhouette_score_average, davies_bouldin_index, calinski_harabasz_sc, clustered_data_file, plot_name


def how_many_days(month, year):
    if month == 2:
        if year == 2016:
            return 29
        return 28
    if year == 2018 and month == 12:
        return 30
    elif month in [4, 6, 9, 11]:
        return 30
    return 31


def flip_table_case1(pod_list, data, month, year, no_days):
    possible_file_path = f'./results/All-pods-{month}-{year}-consume.csv'
    if os.path.exists(possible_file_path):
        df = pd.read_csv(possible_file_path)
        return df
    d = {'POD': []}
    total_consume = []
    initialise = 1
    for pod in pod_list:
        pods = filter_data(data, pod=pod, m1=month, y1=year)
        if not pods.empty:
            d['POD'].append(pod)
            month_consume = 0
            for i in range(1, no_days + 1):
                day_column = f'D{i}'
                month_consume += pods['Consume'].iat[i - 1]
                if initialise:
                    d[day_column] = []
                d[day_column].append(pods['Consume'].iat[i - 1])
            initialise = 0
            total_consume.append('%.2f' % month_consume)
    d['Total Consume'] = total_consume
    df = pd.DataFrame(data=d)
    df.to_csv(f'./results/All-pods-{month}-{year}-consume.csv', index=False)

    return df


def flip_table_case2(data, pod, year):
    possible_file_path = f'./results/{pod}-{year}-consume.csv'
    if os.path.exists(possible_file_path):
        df = pd.read_csv(possible_file_path)
        return df
    d = {'Day': [], 'H1': [], 'H2': [], 'H3': [], 'H4': [], 'H5': [], 'H6': [], 'H7': [], 'H8': [], 'H9': [], 'H10': [],
         'H11': [], 'H12': [], 'H13': [], 'H14': [], 'H15': [], 'H16': [], 'H17': [], 'H18': [], 'H19': [], 'H20': [],
         'H21': [], 'H22': [], 'H23': [], 'H24': []}
    total_consume = []
    line = 0
    for i in range(1, int(len(data) / 24) + 1):  # len(data) / 24 == no_days in that year
        day_consume = 0
        d['Day'].append(f'D{i}')
        for j in range(0, 24):
            day_consume += data['Consume'].iat[line]
            d[f'H{j + 1}'].append(data['Consume'].iat[line])
            line += 1
        total_consume.append('%.2f' % day_consume)

    d['Total Consume'] = total_consume
    df = pd.DataFrame(data=d)
    df.to_csv(f'./results/{pod}-{year}-consume.csv', index=False)

    return df


@csrf_exempt
def call_cluster_case1(request):
    if request.method == 'POST':
        data = pd.read_csv(per_day_path)
        request_data = JSONParser().parse(request)
        algorithm = request_data['algorithm']
        params = request_data['params']
        distances = request_data['distances']
        year = request_data['year']
        month = request_data['month']
        no_days = how_many_days(month, year)

        # To be deleted after data is fixed
        # pod_list = []
        # with open(f"{settings.BASE_DIR}" + r'\results\ok.txt') as f:
        #     for line in f.readlines():
        #         pod_list.append(line[:-1])
        # f.close()
        # --------------------------------
        pod_list = filter_data(data, print_POD_list=True)

        # Data filtered by specified month
        filtered_data = filter_data(data=data, m1=month, y1=year)

        # Result structure
        clustered_data_list = {'Euclidean': {}, 'Manhattan': {},
                               'Cosine': {}}

        # Flip the table
        df = flip_table_case1(pod_list, filtered_data, month, year, no_days)
        if algorithm == 'KMeans':
            # KMeans params
            n_clusters = params['n_clusters']
            clustered_data_list['cluster_list'] = [i for i in range(n_clusters)]
            # Call KMeans
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        k_means(df, n_clusters=n_clusters,
                                date1=datetime.datetime(year, month, 1),
                                euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        k_means(df, n_clusters=n_clusters,
                                date1=datetime.datetime(year, month, 1),
                                manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        k_means(df, n_clusters=n_clusters,
                                date1=datetime.datetime(year, month, 1),
                                cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                else:
                    return JsonResponse("Distance not available", safe=False)
        elif algorithm == 'AHC':
            # AHC params
            n_clusters = params['n_clusters']
            clustered_data_list['cluster_list'] = [i for i in range(n_clusters)]

            # Call AHC
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        ahc(df, n_clusters=n_clusters,
                            date1=datetime.datetime(year, month, 1),
                            euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        ahc(df, n_clusters=n_clusters,
                            date1=datetime.datetime(year, month, 1),
                            manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        ahc(df, n_clusters=n_clusters,
                            date1=datetime.datetime(year, month, 1),
                            cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                else:
                    return JsonResponse("Distance not available", safe=False)

        elif algorithm == 'DBSCAN':
            # DBSCAN params
            eps = params['eps']
            min_samples = params['min_samples']
            clustered_data = {}

            # Call DBSCAN
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        dbscan(df, eps=eps, min_samples=min_samples,
                               date1=datetime.datetime(year, month, 1),
                               euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        dbscan(df, eps=eps, min_samples=min_samples,
                               date1=datetime.datetime(year, month, 1),
                               manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        dbscan(df, eps=eps, min_samples=min_samples,
                               date1=datetime.datetime(year, month, 1),
                               cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                else:
                    return JsonResponse("DIstance not available", safe=True)
            clustered_data_list['cluster_list'] = sorted(list(set(clustered_data['cluster'].tolist())))

        elif algorithm == 'OPTICS':
            # OPTICS params
            eps = params['eps']
            min_samples = params['min_samples']
            clustered_data = {}

            # Call OPTICS
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        optics(df, eps=eps, min_samples=min_samples,
                               date1=datetime.datetime(year, month, 1),
                               euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        optics(df, eps=eps, min_samples=min_samples,
                               date1=datetime.datetime(year, month, 1),
                               manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        optics(df, eps=eps, min_samples=min_samples,
                               date1=datetime.datetime(year, month, 1),
                               cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                else:
                    return JsonResponse("Distance not available", safe=True)
            clustered_data_list['cluster_list'] = sorted(list(set(clustered_data['cluster'].tolist())))
        global case12
        case12 = clustered_data_list
        return JsonResponse(clustered_data_list, safe=False)
    return JsonResponse("Request gresit", safe=False)


@csrf_exempt
def call_cluster_case2(request):
    if request.method == 'POST':
        data = pd.read_csv(csv_path)
        request_data = JSONParser().parse(request)
        algorithm = request_data['algorithm']
        params = request_data['params']
        pod = request_data['pod']
        distances = request_data['distances']
        year = request_data['year']

        df = filter_data(data=data, pod=pod, y1=year)  # Filter data by pod
        df = flip_table_case2(df, pod, year)  # Flip the table

        # Result structure
        clustered_data_list = {'Euclidean': {}, 'Manhattan': {},
                               'Cosine': {}}
        if algorithm == 'KMeans':
            n_clusters = params['n_clusters']  # KMeans params
            clustered_data_list['cluster_list'] = [i for i in range(n_clusters)]

            # Call KMeans
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        k_means(df, n_clusters=n_clusters, pod=pod,
                                date1=datetime.datetime(year, 1, 1),
                                euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        k_means(df, n_clusters=n_clusters, pod=pod,
                                date1=datetime.datetime(year, 1, 1),
                                manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        k_means(df, n_clusters=n_clusters, pod=pod,
                                date1=datetime.datetime(year, 1, 1),
                                cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                else:
                    return JsonResponse("Distance not available", safe=False)
        elif algorithm == 'OPTICS':
            # OPTICS params
            eps = params['eps']
            min_samples = params['min_samples']
            clustered_data = {}

            # Call OPTICS
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        optics(df, eps=eps, min_samples=min_samples, pod=pod,
                               date1=datetime.datetime(year, 1, 1),
                               euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        optics(df, eps=eps, min_samples=min_samples, pod=pod,
                               date1=datetime.datetime(year, 1, 1),
                               manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        optics(df, eps=eps, min_samples=min_samples, pod=pod,
                               date1=datetime.datetime(year, 1, 1),
                               cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                else:
                    return JsonResponse("Distance not available", safe=True)
            clustered_data_list['cluster_list'] = sorted(list(set(clustered_data['cluster'].tolist())))
        elif algorithm == 'AHC':
            # AHC params
            n_clusters = params['n_clusters']
            clustered_data_list['cluster_list'] = [i for i in range(n_clusters)]

            # Call AHC
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        ahc(df, n_clusters=n_clusters, pod=pod,
                            date1=datetime.datetime(year, 1, 1),
                            euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        ahc(df, n_clusters=n_clusters, pod=pod,
                            date1=datetime.datetime(year, 1, 1),
                            manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        ahc(df, n_clusters=n_clusters, pod=pod,
                            date1=datetime.datetime(year, 1, 1),
                            cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                else:
                    return JsonResponse("Distance not available", safe=False)
        elif algorithm == 'DBSCAN':
            # DBSCAN params
            eps = params['eps']
            min_samples = params['min_samples']
            clustered_data = {}

            # Call DBSCAN
            for distance in distances:
                if distance == 'Euclidean':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        dbscan(df, eps=eps, min_samples=min_samples, pod=pod,
                               date1=datetime.datetime(year, 1, 1),
                               euclidean=True)
                    clustered_data_list['Euclidean']['data'] = clustered_data_path
                    clustered_data_list['Euclidean']['plot'] = plot_path
                    clustered_data_list['Euclidean']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]

                elif distance == 'Manhattan':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        dbscan(df, eps=eps, min_samples=min_samples, pod=pod,
                               date1=datetime.datetime(year, 1, 1),
                               manhattan=True)
                    clustered_data_list['Manhattan']['data'] = clustered_data_path
                    clustered_data_list['Manhattan']['plot'] = plot_path
                    clustered_data_list['Manhattan']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                elif distance == 'Cosine':
                    clustered_data, silhouette, davies_bouldin, calinski_harabasz, clustered_data_path, plot_path = \
                        dbscan(df, eps=eps, min_samples=min_samples, pod=pod,
                               date1=datetime.datetime(year, 1, 1),
                               cosine=True)
                    clustered_data_list['Cosine']['data'] = clustered_data_path
                    clustered_data_list['Cosine']['plot'] = plot_path
                    clustered_data_list['Cosine']['table'] = [("Silhouette score", '%.2f' % silhouette, "(-1,1)", "Closer to 1"),
                                                                 ("Davis-Bouldin's index", '%.2f' % davies_bouldin, "[0;inf)", "Low values"),
                                                                 ("Calinski-Harabasz's index", '%.2f' % calinski_harabasz, "[0;inf)", "High values")]
                else:
                    return JsonResponse("Distance not available", safe=True)
            clustered_data_list['cluster_list'] = sorted(list(set(clustered_data['cluster'].tolist())))
        global case12
        case12 = clustered_data_list
        return JsonResponse(clustered_data_list, safe=False)
    return JsonResponse("Request gresit", safe=False)


@csrf_exempt
def one_cluster_view(request):
    if request.method == 'POST':
        request_data = JSONParser().parse(request)
        clustered_data_path = request_data['data']
        if 'All' in clustered_data_path:
            days = True
        else:
            days = False
        cluster = request_data['cluster']
        data = pd.read_csv(clustered_data_path)
        data = data[data['cluster'] == cluster]
        if days:
            specific_cluster_pod_list = filter_data(data, print_POD_list=True)
        else:
            year = int(clustered_data_path.split("/")[1].split("_")[3])
            pod = clustered_data_path.split("/")[1].split("_")[0] + "_" + clustered_data_path.split("/")[1].split("_")[1]
            df = pd.read_csv(per_day_path)
            df = filter_data(df, pod=pod)
            print(df)
            df = filter_data(df, y1=year)
            print(df)
            days_list = data['Day'].tolist()
            specific_cluster_pod_list = []
            for day in range(len(days_list)):
                value = f"{df['Day'].iat[day]}-{month_to_string(df['Month'].iat[day])}-{year}"
                specific_cluster_pod_list.append(value)
        df = data.drop(['cluster', 'Total Consume', 'POD', 'Day'], axis=1, errors='ignore')
        result = {'plot_name': "", 'pod_list': specific_cluster_pod_list}

        # Create line chart
        plt.clf()
        df.T.plot(legend=False)
        plt.ylabel('Energy Consumption(kWh)')
        if days:
            plt.xlabel('Day')
            plt.title(f'Energy Consumption for all PODS in cluster {cluster} in one month')
            plot_name = 'Energy Consumption for one month.png'
            plot_path = f'./media/{plot_name}'
        else:
            plt.xlabel('Hour')
            plt.title(f'Energy Consumption for {pod} in cluster {cluster} in one day')
            plot_name = 'Energy Consumption for one day.png'
            plot_path = f'./media/{plot_name}'
        plt.savefig(plot_path)
        result['plot_name'] = plot_name
        global line
        line = result
        return JsonResponse(result, safe=False)
    return JsonResponse("Request gresit", safe=False)


@csrf_exempt
def consummers_consume(request):
    if request.method == 'POST':
        request_data = JSONParser().parse(request)
        pod = request_data['pod']
        display_type = request_data['display_type']
        result = {"plot_name": ""}
        global consume
        # Select how do we want to display the consume
        if display_type.lower() == 'daily':
            path = csv_path
            day = request_data['day']
            month = request_data['month']
            year = request_data['year']

            data = pd.read_csv(path)
            data_day = filter_data(data=data, pod=pod, d1=day, m1=month, y1=year)

            # Create a bar chart
            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.bar(data_day['Hour'].astype(str), data_day['Consume'])

            # Add labels and title
            plt.xlabel('Hour')
            plt.ylabel('Energy consumption(kWh)')
            plt.title(f'Electrical Consumption per Hour {day}/{month}/{year}: {pod}')

            plot_name = f'Energy consumption per Hour {day}-{month}-{year}-{pod}.png'
            plot_path = f"./media/{plot_name}"
            result['plot_name'] = plot_name

            # Save the plot
            plt.savefig(plot_path)
            consume = result
            return JsonResponse(result, safe=False)
        elif display_type.lower() == 'monthly':
            path = per_day_path
            month = request_data['month']
            year = request_data['year']

            data = pd.read_csv(path)
            data_month = filter_data(data=data, pod=pod, m1=month, y1=year)

            # Create a bar chart
            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.bar(data_month['Day'].astype(str), data_month['Consume'])

            # Add labels and title
            plt.xlabel('Day')
            plt.ylabel('Energy consumption(kWh)')
            plt.title(f'Electrical Consumption per Day {month}/{year}: {pod}')

            plot_name = f'Energy consumption per Day {month}-{year}-{pod}.png'
            plot_path = f'./media/{plot_name}'
            result['plot_name'] = plot_name

            # Save the plot
            plt.savefig(plot_path)
            consume = result
            return JsonResponse(result, safe=False)
        elif display_type.lower() == 'yearly':
            path = per_month_path
            year = request_data['year']

            data = pd.read_csv(path)
            data_year = filter_data(data=data, pod=pod, y1=year)

            # Create a bar chart
            plt.clf()
            plt.figure(figsize=(12, 6))
            plt.bar(data_year['Month'].astype(str), data_year['Consume'])

            # Add labels and title
            plt.xlabel('Month')
            plt.ylabel('Energy consumption(kWh)')
            plt.title(f'Energy consumption per Month {year}: {pod}')

            plot_name = f'Electrical Consumption per Month {year}-{pod}.png'
            plot_path = f'./media/{plot_name}'
            result['plot_name'] = plot_name

            # Save the plot
            plt.savefig(plot_path)
            consume = result
            return JsonResponse(result, safe=False)
        else:
            return JsonResponse("Something went wrong. Please select \'daily\', \'monthly\' or \'yearly\'")
    return JsonResponse("Request gresit", safe=False)


@csrf_exempt
def consume_comparison(request):
    if request.method == 'POST':
        request_data = JSONParser().parse(request)
        pod = request_data['pod']
        m1 = request_data['m1']
        m2 = request_data['m2']
        m3 = request_data['m3']
        m4 = request_data['m4']
        y1 = request_data['y1']
        y2 = request_data['y2']
        y3 = request_data['y3']
        y4 = request_data['y4']

        data = pd.read_csv(per_month_path)
        df1 = filter_data(data, m1=m1, y1=y1, m2=m2, y2=y2, pod=pod)
        df2 = filter_data(data, m1=m3, y1=y3, m2=m4, y2=y4, pod=pod)

        df1['date'] = pd.to_datetime(df1['Month'].astype(str) + '-' + df1['Year'].astype(str))
        df2['date'] = pd.to_datetime(df2['Month'].astype(str) + '-' + df2['Year'].astype(str))

        dates1 = list(df1['date'])
        dates2 = list(df2['date'])

        # Create a bar chart
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.bar(df1['date'], df1['Consume'], color='blue', label='Period 1', width=20)
        plt.bar(df2['date'], df2['Consume'], color='orange', label='Period 2', width=20)

        plt.xticks([date for date, value in
                    zip(list(df1['date']) + list(df2['date']), list(df1['Consume']) + list(df2['Consume'])) if
                    value > 0])
        plt.xticks(rotation=90)

        # Add labels and title
        plt.xlabel('Month')
        plt.ylabel('Energy consumption(kWh)')
        plt.title(f'Consume comparison')
        plt.legend()
        plt.tight_layout()

        plot_name = 'Energy consumption comparison.png'
        plot_path = f'./media/{plot_name}'
        result = {'plot_name': plot_name}

        # Save the plot
        plt.savefig(plot_path)
        global compare
        compare = result
        return JsonResponse(result, safe=False)
    return JsonResponse("Wrong request type!!", safe=False)


def login(request):
    context = {}
    return render(request, 'templates/registration/login.html', context)


def clustering_view(request):

    return render(request, 'backend_templates/cluster.html')


def plot_view(request):
    global case12
    response_data = case12
    # Perform clustering operation and generate image here
    euclidean_plot_url = urljoin(settings.MEDIA_URL,
                                 response_data['Euclidean']['plot']) if 'plot' in response_data['Euclidean'] else None
    manhattan_plot_url = urljoin(settings.MEDIA_URL,
                                 response_data['Manhattan']['plot']) if 'plot' in response_data['Manhattan'] else None
    cosine_plot_url = urljoin(settings.MEDIA_URL,
                              response_data['Cosine']['plot']) if 'plot' in response_data['Cosine'] else None
    euclidean_data_url = 'results/' + response_data["Euclidean"]["data"] if 'data' in response_data["Euclidean"] else None
    manhattan_data_url = 'results/' + response_data["Manhattan"]["data"] if 'data' in response_data["Manhattan"] else None
    cosine_data_url = 'results/' + response_data["Cosine"]["data"] if 'data' in response_data["Cosine"] else None
    print(f"E {euclidean_data_url}")
    print(f"M {manhattan_data_url}")
    print(f"C {cosine_data_url}")
    # returnrender(request, 'backend_templates/plot_page.html', {'data': response_data})
    return render(request, 'backend_templates/plot_page.html', {'euclidean_plot_url': euclidean_plot_url,
                                                                'manhattan_plot_url': manhattan_plot_url,
                                                                'cosine_plot_url': cosine_plot_url,
                                                                'euclidean_data_url': euclidean_data_url,
                                                                'manhattan_data_url': manhattan_data_url,
                                                                'cosine_data_url': cosine_data_url,
                                                                'data': response_data})


def line_chart_view(request):
    global line
    response_data = line
    # Perform clustering operation and generate image here
    image_url = urljoin(settings.MEDIA_URL, response_data['plot_name'])

    return render(request, 'backend_templates/linechart_page.html', {'data': response_data, 'image_url': image_url})


def all_pods_view(request):

    return render(request, 'backend_templates/allPods.html')


def specific_pod_view(request):

    return render(request, 'backend_templates/specificPod.html')


def get_pod_values(request):
    print(settings.BASE_DIR)
    df = pd.read_csv(per_month_path)
    filter_data(df, print_POD_list=True)
    with open(f"{settings.BASE_DIR}" + r'\results\PODs list.txt', 'r') as file:
        data = file.read()
    return HttpResponse(data, content_type='text/plain')


def daily_monthly_yearly_view(request):
    user = request.user
    user_profile = user.userprofile
    assigned_pod = user_profile.assigned_pod

    return render(request, 'backend_templates/daily_monthly_yearly.html', {'assigned_pod': assigned_pod})


def report_view(request):
    global consume
    response_data = consume
    image_url = urljoin(settings.MEDIA_URL, response_data['plot_name'])

    return render(request, 'backend_templates/reports.html', {'image_url': image_url, 'data': response_data})


def compare_view(request):
    user = request.user
    user_profile = user.userprofile
    assigned_pod = user_profile.assigned_pod

    return render(request, 'backend_templates/compare.html', {'assigned_pod': assigned_pod})


def compare_report_view(request):
    global compare
    response_data = compare
    image_url = urljoin(settings.MEDIA_URL, response_data['plot_name'])

    return render(request, 'backend_templates/compare_reports.html', {'image_url': image_url, 'data': response_data})


def registerPage(request):
    form = CreateUserForm()
    if request.method == 'POST':
        form = CreateUserForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_staff = True
            user.is_superuser = True
            user.save()
            return redirect('login')
    context = {'form': form}
    return render(request, 'backend_templates/register.html', context)
