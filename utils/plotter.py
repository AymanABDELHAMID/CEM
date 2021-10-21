import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go
from plotly.offline import iplot

# plotting vizualizer taken from
# https://github.com/faizanahemad/data-science/blob/master/exploration_projects/imbalance-noise-oversampling/lib.py#L28

def visualize_3d(X, y, algorithm="tsne", title="Data in 3D"):
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    if algorithm == "tsne":
        reducer = TSNE(n_components=3, random_state=47, n_iter=400, angle=0.6)
    elif algorithm == "pca":
        reducer = PCA(n_components=3, random_state=47)
    else:
        raise ValueError("Unsupported dimensionality reduction algorithm given.")

    if X.shape[1] > 3:
        X = reducer.fit_transform(X)
    else:
        if type(X) == pd.DataFrame:
            X = X.values

    marker_shapes = ["circle", "diamond", "circle-open", "square", "diamond-open", "cross", "square-open", ]
    traces = []
    for hue in np.unique(y):
        X1 = X[y == hue]

        trace = go.Scatter3d(
            x=X1[:, 0],
            y=X1[:, 1],
            z=X1[:, 2],
            mode='markers',
            name=str(hue),
            marker=dict(
                size=12,
                symbol=marker_shapes.pop(),
                line=dict(
                    width=int(np.random.randint(3, 10) / 10)
                ),
                opacity=int(np.random.randint(6, 10) / 10)
            )
        )
        traces.append(trace)

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title='Dim 1'),
            yaxis=dict(
                title='Dim 2'),
            zaxis=dict(
                title='Dim 3'), ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=traces, layout=layout)
    iplot(fig)