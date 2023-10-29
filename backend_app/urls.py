from django.urls import re_path, path
from . import views

urlpatterns = [
    re_path(f'^consumer/$', views.consumerApi),
    re_path(f'^consumer/([0-9]+)$', views.consumerApi),
    re_path(f'^researcher/$', views.researcherApi),
    re_path(f'^researcher/([0-9]+)$', views.researcherApi),

    re_path(f'^case1/$', views.call_cluster_case1, name='case1'),
    re_path(f'^case2/$', views.call_cluster_case2),

    path('clusteringApp/', views.clustering_view, name='clusteringApp'),
    path('show_plot/', views.plot_view, name='plot_page'),
    path('show_line_chart/', views.line_chart_view, name='line-chart'),
    path('allPods/', views.all_pods_view, name='all-pods'),
    path('specificPod/', views.specific_pod_view, name='specific-pod'),
    path('daily_monthly_yearly/', views.daily_monthly_yearly_view, name='daily_monthly_yearly'),
    path('comparison/', views.compare_view, name='comparison-page'),
    path('comparison_reports/', views.compare_report_view, name='comparison-reports'),
    path('reports/', views.report_view, name='reports'),

    re_path(f'^consume/$', views.consummers_consume),
    re_path(f'^compare/$', views.consume_comparison),
    re_path(f'^cluster/$', views.one_cluster_view),

    path('get_pod_values/', views.get_pod_values),
    path('register/', views.registerPage, name='register'),
]

