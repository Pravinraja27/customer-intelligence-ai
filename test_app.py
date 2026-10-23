"""Quick syntax and import test."""
import sys
import ast
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

files = [
    'src/data_loader.py',
    'src/analytics.py',
    'src/clustering.py',
    'src/prediction.py',
    'src/ai_agent.py',
    'src/visualizations.py',
    'app.py',
]

print("=== SYNTAX CHECK ===")
all_ok = True
for f in files:
    with open(f, encoding='utf-8') as fh:
        try:
            ast.parse(fh.read())
            print(f"  SYNTAX OK: {f}")
        except SyntaxError as e:
            print(f"  SYNTAX ERROR in {f}: {e}")
            all_ok = False

print("\n=== IMPORT CHECK ===")
try:
    from src.data_loader import generate_sample_data, get_data_quality_report, get_numeric_columns
    print("  data_loader imported OK")
    df = generate_sample_data()
    print(f"  Sample data: {len(df)} rows x {len(df.columns)} cols")
    dq = get_data_quality_report(df)
    print(f"  Data quality: {dq['total_records']} records, {dq['missing_values_total']} missing")
except Exception as e:
    print(f"  data_loader FAILED: {e}")
    all_ok = False

try:
    from src.analytics import compute_kpis
    kpis = compute_kpis(df)
    print(f"  analytics OK: total_customers={kpis['total_customers']}, avg_spending={kpis.get('avg_spending_score')}")
except Exception as e:
    print(f"  analytics FAILED: {e}")
    all_ok = False

try:
    from src.clustering import run_kmeans, label_clusters, get_cluster_summary
    r = run_kmeans(df, ['Age', 'AnnualIncome', 'SpendingScore', 'PurchaseFrequency'], 4)
    if r.get('error'):
        print(f"  clustering ERROR: {r['error']}")
        all_ok = False
    else:
        print(f"  clustering OK: silhouette={r['silhouette']:.4f}, labels shape={len(r['labels'])}")
        labeled = label_clusters(df, r['labels'], r['valid_indices'], r['profile'], ['Age','AnnualIncome','SpendingScore','PurchaseFrequency'])
        cs = get_cluster_summary(labeled)
        print(f"  cluster summary: {cs['n_clusters']} clusters")
except Exception as e:
    print(f"  clustering FAILED: {e}")
    all_ok = False

try:
    from src.prediction import get_suitable_features, train_model, get_model_metrics_summary
    feats = get_suitable_features(df, 'SpendingScore')
    print(f"  prediction features: {feats}")
    pm = train_model(df, feats, 'SpendingScore')
    if pm.get('error'):
        print(f"  prediction ERROR: {pm['error']}")
    else:
        print(f"  prediction OK: MAE={pm['mae']}, R2={pm['r2']}, train={pm['n_train']}, test={pm['n_test']}")
except Exception as e:
    print(f"  prediction FAILED: {e}")
    all_ok = False

try:
    from src.visualizations import income_vs_spending_scatter, gender_pie, correlation_heatmap
    f1 = income_vs_spending_scatter(df)
    f2 = gender_pie(df)
    f3 = correlation_heatmap(df)
    print(f"  visualizations OK: scatter={f1 is not None}, pie={f2 is not None}, heatmap={f3 is not None}")
except Exception as e:
    print(f"  visualizations FAILED: {e}")
    all_ok = False

try:
    from src.ai_agent import respond, SUGGESTED_QUESTIONS, _openai_available
    state = {'df': df, 'kpis': kpis, 'cluster_summary': cs if 'cs' in dir() else None, 'pred_result': pm if 'pm' in dir() else None, 'data_quality': dq}
    questions = ["Summarize the customer dataset.", "Explain the customer segments.", "What strategies could improve customer retention?"]
    for q in questions:
        resp, mode = respond(q, state, [])
        print(f"  ai_agent [{mode}]: Q='{q[:40]}' -> {len(resp)} chars")
    print(f"  OpenAI available: {_openai_available}")
except Exception as e:
    print(f"  ai_agent FAILED: {e}")
    all_ok = False

print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
