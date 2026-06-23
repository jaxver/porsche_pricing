import pandas as pd
import os
from datetime import datetime
from pathlib import Path

from elferspot_listings.utils.dashboard_data import (
    load_latest_benchmark_outputs,
)

DATA_PATH = os.path.join('data', 'all_listings_gold.xlsx')
BENCHMARK_RESULTS_PATH = Path('results') / 'benchmarks'

def load_data(path=DATA_PATH):
    if path.lower().endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # Basic cleaning
    df = df.drop_duplicates(subset=['URL']) if 'URL' in df.columns else df
    # Ensure price_in_eur exists or compute if price & currency present
    if 'price_in_eur' not in df.columns and 'price' in df.columns and 'currency' in df.columns:
        df['price_in_eur'] = df['price']  # assume already converted or preprocessed
    # Parse date fields if present
    for col in ['Scraped_At', 'Scraped At', 'Date']:
        if col in df.columns:
            df['scraped_at'] = pd.to_datetime(df[col], errors='coerce')
            break
    if 'scraped_at' not in df.columns:
        df['scraped_at'] = pd.NaT
    # Year fallback
    if 'Year of construction' in df.columns:
        df['year'] = pd.to_numeric(df['Year of construction'], errors='coerce')
    else:
        df['year'] = pd.NA
    # Mileage numeric
    for col in ['Mileage_km', 'Mileage', 'mileage_value']:
        if col in df.columns:
            df['mileage_km'] = pd.to_numeric(df[col], errors='coerce')
            break
    if 'mileage_km' not in df.columns:
        df['mileage_km'] = pd.NA
    return df


def _metrics_to_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for model_name, values in metrics.items():
        if isinstance(values, dict):
            row = {'model_name': model_name, **values}
        else:
            row = {'model_name': model_name, 'value': values}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    import plotly.express as px
    import streamlit as st

    st.set_page_config(layout='wide', page_title='Porsche Listings Browser')
    st.title('Porsche Listings Dashboard')

    benchmark_outputs = load_latest_benchmark_outputs(BENCHMARK_RESULTS_PATH)

    st.subheader('Latest benchmark run')
    if benchmark_outputs is None:
        st.info('No benchmark predictions found under results/benchmarks.')
    else:
        st.caption(f'Loaded from {benchmark_outputs.run_dir}')

        st.markdown('**Metrics**')
        if benchmark_outputs.metrics:
            if isinstance(benchmark_outputs.metrics, dict):
                metrics_frame = _metrics_to_frame(benchmark_outputs.metrics)
                st.dataframe(metrics_frame, use_container_width=True)
            else:
                st.json(benchmark_outputs.metrics)
        else:
            st.info('No metrics.json found for the latest benchmark run.')

        st.markdown('**Predictions**')
        predictions = benchmark_outputs.predictions
        if predictions is not None and not predictions.empty:
            prediction_columns = [
                column
                for column in ['row_index', 'model_name', 'actual_price_eur', 'predicted_price_eur', 'residual_eur']
                if column in predictions.columns
            ]
            display_predictions = predictions[prediction_columns] if prediction_columns else predictions
            st.dataframe(display_predictions.head(200), use_container_width=True)
        else:
            st.info('No predictions.csv found for the latest benchmark run.')

    load_data_cached = st.cache_data(load_data)
    df = load_data_cached()

    # Sidebar filters
    st.sidebar.header('Filters')
    models = sorted(df['Model'].dropna().unique()) if 'Model' in df.columns else []
    selected_models = st.sidebar.multiselect('Model', models, default=models[:10])

    # Gearbox / Transmission filter
    gearbox_col_candidates = [c for c in df.columns if 'gear' in c.lower() or 'transm' in c.lower() or 'transmission' in c.lower()]
    if gearbox_col_candidates:
        gearbox_col = gearbox_col_candidates[0]
        gearboxes = sorted(df[gearbox_col].dropna().unique())
        selected_gearboxes = st.sidebar.multiselect('Gearbox / Transmission', gearboxes, default=gearboxes[:10])
    else:
        gearbox_col = None
        selected_gearboxes = []

    price_min, price_max = int(df['price_in_eur'].min()) if 'price_in_eur' in df.columns else 0, int(df['price_in_eur'].max()) if 'price_in_eur' in df.columns else 100000
    price_range = st.sidebar.slider('Price (EUR)', price_min, price_max, (price_min, price_max))

    mileage_min = int(df['mileage_km'].min()) if pd.api.types.is_numeric_dtype(df['mileage_km']) else 0
    mileage_max = int(df['mileage_km'].max()) if pd.api.types.is_numeric_dtype(df['mileage_km']) else 200000
    mileage_range = st.sidebar.slider('Mileage (km)', mileage_min, mileage_max, (mileage_min, mileage_max))

    year_min = int(df['year'].min()) if pd.api.types.is_numeric_dtype(df['year']) else 1900
    year_max = int(df['year'].max()) if pd.api.types.is_numeric_dtype(df['year']) else datetime.now().year
    year_range = st.sidebar.slider('Year', year_min, year_max, (year_min, year_max))

    text_search = st.sidebar.text_input('Search Title/Description')

    # Apply filters
    mask = pd.Series(True, index=df.index)
    if selected_models:
        mask &= df['Model'].isin(selected_models)
    if gearbox_col and selected_gearboxes:
        mask &= df[gearbox_col].isin(selected_gearboxes)
    if 'price_in_eur' in df.columns:
        mask &= df['price_in_eur'].between(price_range[0], price_range[1], inclusive='both')
    if pd.api.types.is_numeric_dtype(df['mileage_km']):
        mask &= df['mileage_km'].between(mileage_range[0], mileage_range[1], inclusive='both')
    if pd.api.types.is_numeric_dtype(df['year']):
        mask &= df['year'].between(year_range[0], year_range[1], inclusive='both')
    if text_search:
        txt = text_search.lower()
        cols = [c for c in df.columns if 'desc' in c.lower() or 'title' in c.lower() or 'model' in c.lower()]
        if cols:
            mask &= df[cols].fillna('').apply(lambda r: txt in ' '.join(map(str, r.values)).lower(), axis=1)

    filtered = df[mask]

    st.sidebar.write(f'Listings matching filters: {len(filtered):,}')

    # Main layout
    st.subheader('Listings table')
    display_cols = ['Title', 'Model', 'year', 'mileage_km', 'price_in_eur', 'URL']
    display_cols = [c for c in display_cols if c in filtered.columns]
    if gearbox_col and gearbox_col not in display_cols:
        display_cols.insert(3, gearbox_col)
    st.dataframe(filtered[display_cols].head(500), use_container_width=True)

    # Price over time plot
    st.subheader('Price over time')
    if 'scraped_at' in filtered.columns and not filtered['scraped_at'].isna().all():
        fig = px.scatter(filtered.sort_values('scraped_at'), x='scraped_at', y='price_in_eur', color='Model', hover_data=['Title','URL'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No scraped_at/date column found for time series.')

    # Mileage and price summary
    st.subheader('Mileage and Price summary by Model')
    if 'Model' in filtered.columns:
        agg = filtered.groupby('Model').agg(
            count=('price_in_eur','count'),
            price_mean=('price_in_eur','mean'),
            price_min=('price_in_eur','min'),
            price_max=('price_in_eur','max'),
            mileage_mean=('mileage_km','mean'),
            mileage_min=('mileage_km','min'),
            mileage_max=('mileage_km','max')
        ).reset_index()
        fig2 = px.scatter(agg, x='mileage_mean', y='price_mean', size='count', hover_data=['Model','count','price_min','price_max'])
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(agg.sort_values('count', ascending=False).head(200), use_container_width=True)
    else:
        st.info('No Model column present to aggregate.')

    # Detail view: click to open URL
    st.subheader('Selected listing detail')
    if not filtered.empty:
        idx = st.number_input('Row index (position in filtered DF)', min_value=0, max_value=len(filtered)-1, value=0)
        row = filtered.reset_index(drop=True).iloc[idx]
        st.write(row.to_dict())
        if 'URL' in row and pd.notna(row['URL']):
            st.markdown(f"[Open Listing]({row['URL']})")

if __name__ == '__main__':
    main()
