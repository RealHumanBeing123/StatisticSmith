"""Every chart type listed in the Datylon article (https://www.datylon.com/blog/types-of-charts-graphs-examples-data-visualization)
must exist in the registry under the same category as the article's section."""
from statsmith.charts import REGISTRY

ARTICLE = {
    "Comparison": ["bar", "column", "grouped_column", "grouped_bar", "lollipop", "bullet", "dot_plot", "dumbbell", "pictogram", "icon_chart",
                   "range_plot", "radial_bar", "parallel_coordinates", "radar", "nightingale", "waterfall", "matrix_chart", "small_multiples",
                   "word_cloud", "slope_chart", "table_chart", "categorical_scatter", "quadrant_chart"],
    "Correlation": ["heatmap", "bubble", "scatter", "connected_scatter", "hexbin", "contour"],
    "Part-to-whole & hierarchical": ["stacked_bar", "stacked_column", "diverging_bar", "diverging_column", "population_pyramid", "icon_array", "waffle",
                                     "pie", "donut", "semicircle_donut", "marimekko", "treemap", "circular_treemap", "convex_treemap", "dendrogram",
                                     "venn", "euler", "circular_gauge", "sunburst", "funnel", "pyramid"],
    "Data over time": ["area", "stacked_area", "stream_graph", "bump_chart", "bump_area", "line", "spline", "step_line", "candlestick", "gantt",
                       "barcode", "ohlc"],
    "Distribution": ["density", "ridgeline", "horizon", "histogram", "radial_histogram", "strip", "jitter", "one_d_heatmap", "beeswarm", "box", "violin"],
    "Geospatial & other": ["geo_heatmap", "choropleth", "tile_map", "chord_diagram", "arc_diagram", "sankey", "network_diagram", "flowchart"],
}


def test_all_article_charts_present_in_right_category():
    for cat, ids in ARTICLE.items():
        for i in ids:
            assert i in REGISTRY, f"missing chart {i}"
            assert REGISTRY[i].category == cat, (i, REGISTRY[i].category, cat)


def test_user_named_charts():
    for i in ("histogram", "bar", "pareto", "scatter", "treemap", "box", "line", "heatmap", "pie"):
        assert i in REGISTRY
