from kedro.pipeline import Pipeline, node, pipeline
from .nodes import ConformData, data_conformada, x_y, y_test, svc, pegar_metricas, pycaret_mlsflow, classificacao_pycaret, dimData, filtro_shortType, filtro_shortType3pt, shortType2pt, shortType3pt, regressaoLogistic

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
                func=data_conformada,
                inputs=None,
                outputs='comforme_data',
                name='data_conformada',
        ),
        node(
                func=x_y,
                inputs='comforme_data',
                outputs='x_test',
                name='x_y',
        ),
        node(
                func=y_test,
                inputs='comforme_data',
                outputs='y_teste',
                name='y_test',
        ),
        node(
                func=svc,
                inputs=['x_test', 'y_teste'],
                outputs=['xtrain', 'ytrain'],
                name='svc',
        ),
        node(
                func=pegar_metricas,
                inputs=['xtrain', 'ytrain'],
                outputs=['metrics0', 'selectd_metricsy'],
                name='metrics0',
        ),
        node(
                func=pycaret_mlsflow,
                inputs='conforme_data',
                outputs='selectd_algoritmo_classi',
                name='algoritimos_classificacao',
        ),
        node(
                func=classificacao_pycaret,
                inputs='conforme_data',
                outputs='selectd_algoritmo_regre',
                name='algoritimos_regressao',
        ),
        node(
                func=dimData,
                inputs='conforme_data',
                outputs='dime_dataset',
                name='dimensao_dataset',
        ),
        node(
                func=filtro_shortType,
                inputs=None,
                outputs='filtro_shortType',
                name='filtro_shortType',
        ),
        node(
                func=filtro_shortType3pt,
                inputs=None,
                outputs='filtro_shortType3pt',
                name='filtro_shortType3pt',
        ),
        node(
                func=shortType2pt,
                inputs=None,
                outputs='shortType2pt',
                name='shortType2pt',
        ),
        node(
                func=shortType3pt,
                inputs=None,
                outputs='shortType3pt',
                name='shortType3pt',
        ),
        node(
                func=ConformData,
                inputs=None,
                outputs='dados_limpos',
                name='comforme_data',
        ),
        node(
                func=regressaoLogistic,
                inputs='xtrain',
                outputs='regre_logist',
                name='regressao_logist',
        ),
    ])
