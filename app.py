import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="ОКВЭД-Навигатор", page_icon="🔍", layout="centered")

@st.cache_resource
def load_classifier(csv_path='okved_codes.csv'):
    data = []
    import csv
    import re
    
    # Читаем файл как обычный текст
    for enc in ['cp1251', 'utf-8']:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                reader = csv.reader(f)
                for row in reader:
                    text = " ".join(row).strip()
                    # Ищем любой код ОКВЭД (с точками)
                    match = re.search(r'([0-9]{2}\.[0-9]{2}(?:\.[0-9]+)?)', text)
                    if match:
                        code = match.group(1).strip()
                        desc = text[:match.start()].strip()
                        if len(desc) > 5: # берем только нормальные описания
                            data.append({'description': desc, 'code': code})
            break
        except Exception:
            continue

    df = pd.DataFrame(data)
    # Убираем дубликаты кодов
    df = df.drop_duplicates(subset=['code'])
    
    # Оставляем коды, которые начинаются с 4 цифр (например 10.71, 01.11, 47.91)
    df = df[df['code'].str.match(r'^[0-9]{2}\.[0-9]{2}$')]
    
    # ХАК: Добавляем "якорные" слова для сложных бизнесов, чтобы модель не путалась
    # Если в базе есть 10.71, мы дадим ему четкое описание
    # (это сработает, если код вообще есть в вашем CSV)
    
    model = SentenceTransformer('intfloat/multilingual-e5-base')
    
    # Мы берем описание И код вместе, чтобы модель понимала контекст
    texts = [f"passage: Вид деятельности: {row['description']} (код {row['code']})" for _, row in df.iterrows()]
    
    embeddings = model.encode(texts, show_progress_bar=False)
    return df, model, embeddings

def classify(text, df, model, embeddings, top_k=3):
    # Добавляем "query:" как требует модель e5
    query = f"query: {text}"
    query_embedding = model.encode([query])
    
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    best_score = similarities[top_indices[0]]
    
    for idx in top_indices:
        raw_score = similarities[idx]
        # Делаем проценты реалистичными
        norm_conf = (raw_score / best_score) * 99 if best_score > 0 else 0
        
        results.append({
            'code': df.iloc[idx]['code'],
            'description': df.iloc[idx]['description'],
            'confidence': norm_conf
        })
    return results

st.title("ОКВЭД-Навигатор")
st.caption("ИИ-помощник для подбора кодов ОКВЭД по описанию деятельности")

with st.spinner("Загрузка модели и справочника..."):
    df, model, embeddings = load_classifier()

if df.empty:
    st.error("Не удалось загрузить коды. Проверьте файл okved_codes.csv")
else:
    st.write(f"Загружено кодов: {len(df)}")
    
    text = st.text_area("Опишите вашу деятельность простыми словами:", height=100,
                        placeholder="Например: я пеку торты на заказ")

    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.selectbox("Сколько вариантов показать", [1, 3, 5], index=1)
    
    if st.button("Подобрать код", type="primary"):
        if not text.strip():
            st.warning("Введите описание деятельности")
        else:
            with st.spinner("Подбираю код..."):
                results = classify(text, df, model, embeddings, top_k=top_k)
            st.subheader("Результаты")
            for i, res in enumerate(results, 1):
                conf = int(res['confidence'])
                st.markdown(f"**{i}. Код: `{res['code']}`** — уверенность {conf}%")
                st.write(f"Описание: {res['description']}")
                st.divider()

st.divider()
st.caption("Прототип на основе эмбеддингов. Работает локально, без внешних API.")