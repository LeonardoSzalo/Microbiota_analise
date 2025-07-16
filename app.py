from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64
import google.generativeai as genai
import os
from io import BytesIO 
from Bio import Entrez 
from skbio.stats.ordination import pcoa 
from skbio import DistanceMatrix 
from dotenv import load_dotenv
load_dotenv()

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Variáveis alvo que o modelo irá predizer.
TARGET_VARIABLES = ["age_months", "body_weight"] 

#CHAVES APIs
DEV_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEV_NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_EMAIL = os.getenv("NCBI_EMAIL")

_gemini_api_configured_successfully = False
_ncbi_api_configured_successfully = False 


# Cria o diretório de uploads se ele não existir
if not os.path.exists(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER'])):
    os.makedirs(os.path.join(os.getcwd(), app.config['UPLOAD_FOLDER']))


def configure_apis_global():
    """
    Configura as APIS do Gemini e do NCBI
    """
    global _gemini_api_configured_successfully
    global _ncbi_api_configured_successfully 

    print("DEBUG: Tentando configurar a API do Gemini...")
    if DEV_GEMINI_API_KEY and DEV_GEMINI_API_KEY != "chave":
        try:
            genai.configure(api_key=DEV_GEMINI_API_KEY)
            _gemini_api_configured_successfully = True
            print("API do Gemini configurada globalmente com sucesso.")
        except Exception as e:
            _gemini_api_configured_successfully = False
            print(f"ERRO CRÍTICO: Falha ao configurar a API do Gemini globalmente: {e}")
            print("Verifique se a DEV_GEMINI_API_KEY é válida e se há conexão com a internet.")
    else:
        _gemini_api_configured_successfully = False
        print("AVISO: DEV_GEMINI_API_KEY não foi definida ou é o placeholder.")
        print("A funcionalidade de insight da IA (Gemini) pode não estar disponível.")

    print("DEBUG: Tentando configurar a API do NCBI (Entrez)...")
    Entrez.email = NCBI_EMAIL 
    if Entrez.email == "email":
        print("AVISO: NCBI_EMAIL não foi definido. A API Entrez usará o limite de requisições padrão (mais baixo).")
    
    if DEV_NCBI_API_KEY and DEV_NCBI_API_KEY != "chave_ncbi":
        Entrez.api_key = DEV_NCBI_API_KEY
        _ncbi_api_configured_successfully = True
        print("API do NCBI (Entrez) configurada globalmente com sucesso com chave de API.")
    else:
        _ncbi_api_configured_successfully = False 
        print("AVISO: DEV_NCBI_API_KEY não foi definida ou é o placeholder. A API Entrez usará o limite de requisições padrão (mais baixo).")

def load_data_from_memory(file_storage):
    """
    Carrega os dados e faz
    """
    try:
        file_content = file_storage.read()
        file_stream = BytesIO(file_content)

        filename = file_storage.filename
        
        df = None 

        try:
            df = pd.read_csv(file_stream) 
            file_stream.seek(0) 

            if len(df.columns) == 1 and ',' not in df.columns[0]: 
                file_stream.seek(0) 
                df_semicolon_try = pd.read_csv(file_stream, delimiter=';')
                file_stream.seek(0) 
                if len(df_semicolon_try.columns) > 1: 
                    df = df_semicolon_try
                    print("DEBUG: CSV lido com sucesso usando ponto e vírgula como delimitador.")
                else:
                    print("DEBUG: CSV lido com vírgula (default).")
            else:
                print("DEBUG: CSV lido com vírgula (default).")

        except Exception as csv_error:
            print(f"DEBUG: Falha na leitura como CSV. Erro: {csv_error}")
            df = None 
        
        if df is None or df.empty: 
            file_stream.seek(0) 
            if filename.endswith('.xlsx'):
                df = pd.read_excel(file_stream, engine='openpyxl')
                print("DEBUG: Arquivo lido como XLSX.")
            elif filename.endswith('.xls'):
                df = pd.read_excel(file_stream, engine='xlrd')
                print("DEBUG: Arquivo lido como XLS.")
            else:
                raise ValueError("Formato de arquivo não suportado ou erro na leitura CSV. Por favor, use CSV, XLSX ou XLS.")

        if df.empty:
            raise ValueError("O arquivo está vazio ou não pôde ser lido após todas as tentativas.")

        return df, None
    except Exception as e:
        return None, str(e)

def calculate_alpha_diversity(microbiota_data):
    """
    Calcula o índice de Shannon
    """
    microbiota_data = microbiota_data.fillna(0).astype(float)
    proportions = microbiota_data.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)
    proportions[proportions == 0] = np.finfo(float).eps 
    alpha_diversity = entropy(proportions, base=np.e, axis=1)
    return pd.Series(alpha_diversity, index=microbiota_data.index)

def generate_pubmed_query_for_bacteria(top_3_bacteria):
    """
    Gera uma consulta de busca otimizada para o PubMed com base nas 3 espécies mais abundantes, otimizada por IA.
    """
    if not top_3_bacteria: 
        return "" 

    bacteria_names_clean = [bacter_name.replace("s__", "").replace("_", " ") for bacter_name in top_3_bacteria]
    
    prompt = (f"Crie uma consulta de busca formatada para o PubMed. "
              f"A busca deve ser sobre o papel e a importância das seguintes bactérias na microbiota intestinal humana: "
              f"{', '.join(bacteria_names_clean)}. "
              f"Inclua sinônimos e termos relacionados para cada bactéria. "
              f"Use operadores booleanos (AND, OR) e sufixos de campo PubMed ([MeSH], [Title/Abstract]) para otimizar a busca. "
              f"Também inclua termos gerais como 'gut microbiota', 'intestinal health', 'human', 'immune system', 'metabolism'. "
              f"Priorize a abrangência para encontrar artigos relevantes. "
              f"A saída deve ser APENAS a string da consulta, sem explicações ou formatação adicional. "
              f"Exemplo: `(\"Bacteroides\"[MeSH] OR \"Bacteroides\"[Title/Abstract] OR \"Bacteroidetes\"[MeSH]) AND (\"gut microbiota\"[Title/Abstract] OR \"intestinal health\"[Title/Abstract])`"
              f"Sua consulta:")
    
    try:
        model = genai.GenerativeModel() 
        response = model.generate_content(prompt)
        pubmed_query = response.text.strip()
        
        pubmed_query = pubmed_query.strip('`" ') 

        return pubmed_query
    except Exception as e:
        print(f"DEBUG: Erro ao gerar consulta PubMed com Gemini: {e}") 
        return "" 

def search_pubmed_and_get_summaries(query, max_articles=5):
    """
    Busca artigos no PubMed usando a API Entrez e retorna os resumos dos artigos
    """
    global _ncbi_api_configured_successfully

    if not _ncbi_api_configured_successfully and Entrez.email == "email":
        print("AVISO: API Entrez não configurada (email ou chave inválida). Não será possível buscar no PubMed.")
        return [] 

    if not query:
        return []

    article_summaries = []
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_articles, retmode="xml")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record["IdList"]
        print(f"DEBUG: Encontrados {len(id_list)} artigos no PubMed para a consulta.")

        if not id_list:
            return []

        handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml") 
        articles = Entrez.read(handle)
        handle.close()

        for article in articles["PubmedArticle"]:
            try:
                title = article["MedlineCitation"]["Article"]["ArticleTitle"]
                abstract_text = ""
                if "Abstract" in article["MedlineCitation"]["Article"]:
                    abstract_parts = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]
                    if isinstance(abstract_parts, list):
                        abstract_text = " ".join(abstract_parts)
                    else:
                        abstract_text = abstract_parts 
                
                summary_entry = f"Título: {title}\nResumo: {abstract_text}\n"
                if abstract_text.strip(): 
                    article_summaries.append(summary_entry)
                else:
                    print(f"DEBUG: Resumo vazio para o artigo: {title}. Pulando.")

            except KeyError:
                print("DEBUG: Título ou Resumo não encontrado em estrutura esperada para um artigo. Pulando.")
                continue
            except Exception as extract_error:
                print(f"DEBUG: Erro ao extrair resumo/título de um artigo: {extract_error}")
                continue

    except Exception as e:
        print(f"DEBUG: Erro na busca ou fetch do PubMed via Entrez: {e}")
        return []

    return article_summaries

def summarize_articles_or_knowledge_with_gemini(article_texts, top_3_bacteria_names, num_articles_to_summarize=5):
    """
    Tenta resumir artigos fornecidos com Gemini. Se falhar ou não houver artigos, gera um resumo com base no conhecimento prévio do Gemini.
    """
    global _gemini_api_configured_successfully 

    if not _gemini_api_configured_successfully:
        return "API do Gemini não configurada globalmente. Não foi possível gerar resumo."
    
    model = genai.GenerativeModel() 

    if article_texts:
        combined_text = "\n\n---\n\n".join(article_texts[:num_articles_to_summarize])
        if len(combined_text) > 15000: 
            combined_text = combined_text[:15000] + "..." 
        
        try:
            prompt_from_articles = (
                f"Resuma os principais achados e a relevância biológica destes textos de artigos científicos sobre microbiota intestinal. "
                f"Foque nos papéis das bactérias predominantes ({', '.join(top_3_bacteria_names)}) e no contexto de saúde humana. "
                f"O resumo deve ter aproximadamente 5-7 linhas e ser conciso. Não inclua introdução ou conclusão. "
                f"Use **apenas informações contidas nos textos fornecidos** e cite o artigo, se possível, de forma concisa. Exemplo: (Autor, Ano).\n\n"
                f"Textos dos Artigos:\n{combined_text}"
            )
            response = model.generate_content(prompt_from_articles)
            return response.text
        except Exception as e:
            print(f"DEBUG: Erro ao tentar resumir artigos: {e}. Tentando resumo baseado em conhecimento prévio.")
            return summarize_from_knowledge_gemini(top_3_bacteria_names)
    else:
        return summarize_from_knowledge_gemini(top_3_bacteria_names)

def summarize_from_knowledge_gemini(top_3_bacteria):
    """
    Gera um resumo sobre as bactérias com base no conhecimento prévio do Gemini.
    """
    global _gemini_api_configured_successfully 

    if not _gemini_api_configured_successfully:
        return "API do Gemini não configurada para gerar resumo baseado em conhecimento prévio."
    
    if not top_3_bacteria:
        return "Não foi possível identificar bactérias para gerar um insight baseado em conhecimento prévio."

    try:
        model = genai.GenerativeModel() 
        prompt_from_knowledge = (
            f"Com base na literatura científica bem documentada, descreva o papel e a importância das seguintes bactérias predominantes na microbiota intestinal humana: "
            f"{', '.join(top_3_bacteria)}. "
            f"O resumo deve ser bem curto, com uma visão geral de aproximadamente 3-5 linhas. "
            f"Não inclua introdução ou conclusão. Foque no papel geral e importância para a saúde."
        )
        response = model.generate_content(prompt_from_knowledge)
        return response.text
    except Exception as e:
        print(f"DEBUG: Erro ao gerar resumo baseado em conhecimento prévio: {e}")
        return f"Erro ao gerar resumo (conhecimento prévio): {e}. Verifique a API Gemini."


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'reference_db' not in request.files or 'target_sample' not in request.files:
        return render_template('results.html', error="Por favor, envie ambos os arquivos: Base de Dados de Referência e Amostra Alvo.")

    reference_file = request.files['reference_db']
    target_file = request.files['target_sample']

    if reference_file.filename == '' or target_file.filename == '':
        return render_template('results.html', error="Nenhum arquivo selecionado. Por favor, selecione ambos os arquivos.")

    reference_db, error = load_data_from_memory(reference_file)
    if error:
        return render_template('results.html', error=f"Erro ao carregar a base de dados de referência: {error}")

    target_sample, error = load_data_from_memory(target_file)
    if error:
        return render_template('results.html', error=f"Erro ao carregar a amostra alvo: {error}")


    # Identifica as colunas de espécies na base de dados de referência
    species_columns = [col for col in reference_db.columns if col not in TARGET_VARIABLES and col != 'microbial_age'] 
    
    if not species_columns:
        return render_template('results.html', error="Nenhuma coluna de espécie bacteriana identificada na base de dados de referência. Verifique se as TARGET_VARIABLES estão corretas ou se o arquivo está vazio.")

    for target_var in TARGET_VARIABLES:
        if target_var not in reference_db.columns:
            return render_template('results.html', error=f"A variável alvo '{target_var}' não foi encontrada na base de dados de referência. Verifique a lista TARGET_VARIABLES.")
            
    if len(reference_db) < 2:
        return render_template('results.html', error="A base de dados de referência deve conter pelo menos 2 linhas (amostras) para treinar o modelo.")

    X_ref = reference_db[species_columns]
    y_ref = reference_db[TARGET_VARIABLES] 

    if "age_months" not in reference_db.columns:
        return render_template('results.html', error="A coluna 'age_months' é essencial para o cálculo do MAZ e não foi encontrada na base de dados de referência.")


    #Alinha a amostra alvo com as colunas de espécies da base de dados de referência
    target_sample_aligned = target_sample.reindex(columns=species_columns, fill_value=0)

    if target_sample_aligned.empty or target_sample_aligned.isnull().all().all():
        return render_template('results.html', error="Amostra alvo não contém dados compatíveis com as espécies da base de referência após o alinhamento. Verifique os nomes das colunas de espécies ou se o arquivo está vazio.")
    
    if len(target_sample_aligned) != 1:
        return render_template('results.html', error="O arquivo da amostra alvo deve conter apenas uma linha de dados de um único indivíduo.")

    # Modelos
    predictions = {}
    performance_metrics = {}

    for target in TARGET_VARIABLES:
        y_target = y_ref[target]
        
        if len(X_ref) < 2: 
            return render_template('results.html', error=f"Dados insuficientes na base de referência para treinar o modelo para '{target}'. Precisa de pelo menos 2 amostras.")

        test_size = 0.2
        if len(X_ref) * test_size < 1: 
            test_size = 1 / len(X_ref) if len(X_ref) > 1 else 0 
            if test_size == 0: 
                return render_template('results.html', error=f"Dados insuficientes para dividir em treino/teste para '{target}'.")


        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_ref, y_target, test_size=test_size, random_state=42
            )
        except ValueError as e:
            return render_template('results.html', error=f"Erro ao dividir os dados para '{target}': {e}. Provavelmente, dados insuficientes ou coluna alvo com valores constantes.")
            
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        try:
            model.fit(X_train, y_train)
        except ValueError as e:
            return render_template('results.html', error=f"Erro ao treinar o modelo para '{target}': {e}. Verifique se há variância nos dados ou se há amostras suficientes.")

        y_pred_test = model.predict(X_test)
        
        r2 = None
        mae = None
        if len(y_test) > 1 and len(y_test.unique()) > 1: 
            try:
                r2 = r2_score(y_test, y_pred_test)
            except Exception as e:
                print(f"DEBUG: R2 calculation failed for {target}: {e}")
                r2 = None 
        else:
            print(f"DEBUG: R2 not calculated for {target} due to insufficient test samples or constant values.")

        if len(y_test) > 0: 
            try:
                mae = mean_absolute_error(y_test, y_pred_test)
            except Exception as e:
                print(f"DEBUG: MAE calculation failed for {target}: {e}")
                mae = None

        performance_metrics[target] = {"R2": r2, "MAE": mae}
        
        prediction = model.predict(target_sample_aligned)
        predictions[target] = prediction[0]


    microbial_age_predictor_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    if "age_months" not in reference_db.columns:
        return render_template('results.html', error="A coluna 'age_months' é essencial na base de referência para treinar o modelo que calcula a 'microbial_age' (idade predita da microbiota) e, consequentemente, o MAZ.")
    
    y_age_months_ref_for_microbial_age_model = reference_db["age_months"]
    
    if len(X_ref) < 2:
        return render_template('results.html', error="Dados insuficientes para treinar o modelo que calcula 'microbial_age' a partir da base de referência (requer pelo menos 2 amostras).")

    try:
        microbial_age_predictor_model.fit(X_ref, y_age_months_ref_for_microbial_age_model)
    except ValueError as e:
        return render_template('results.html', error=f"Erro ao treinar o modelo para calcular 'microbial_age' (baseado em age_months): {e}. Verifique a variância em 'age_months' ou a quantidade de amostras na base de referência.")

    predicted_microbial_ages_ref_array = microbial_age_predictor_model.predict(X_ref)
    
    mediana_microbiana_ref = np.median(predicted_microbial_ages_ref_array)
    desvio_padrao_microbiano_ref = np.std(predicted_microbial_ages_ref_array)

    predicted_microbial_age_target = microbial_age_predictor_model.predict(target_sample_aligned)[0] 
    
    maz_value = None
    if predicted_microbial_age_target is not None:
        if desvio_padrao_microbiano_ref > 0:
            maz_value = (predicted_microbial_age_target - mediana_microbiana_ref) / desvio_padrao_microbiano_ref
        else:
            maz_value = 0.0 if predicted_microbial_age_target == mediana_microbiana_ref else None 
    else:
        return render_template('results.html', error="Não foi possível obter a 'microbial_age' predita para o indivíduo alvo para calcular o MAZ.")


    # Alfa Diversidade do Indivíduo Alvo (Shannon Index)
    alpha_diversity_value = None
    try:
        target_species_data = target_sample_aligned[species_columns]
        alpha_diversity_series = calculate_alpha_diversity(target_species_data)
        alpha_diversity_value = alpha_diversity_series.iloc[0] if not alpha_diversity_series.empty else None
    except Exception as e:
        print(f"Erro ao calcular alfa diversidade: {e}")
        alpha_diversity_value = None

    # Cálculo da média e desvio padrão do índice de Shannon da base de referência
    alpha_diversity_ref_mean = None
    alpha_diversity_ref_std = None
    try:
        ref_species_data = X_ref[species_columns] 
        if not ref_species_data.empty:
            ref_alpha_diversities = calculate_alpha_diversity(ref_species_data)
            alpha_diversity_ref_mean = ref_alpha_diversities.mean()
            alpha_diversity_ref_std = ref_alpha_diversities.std()
    except Exception as e:
        print(f"Erro ao calcular média/std da alfa diversidade da referência: {e}")
        alpha_diversity_ref_mean = None
        alpha_diversity_ref_std = None


    # Beta Diversidade (Bray-Curtis) PCoA Plot 
    pcoa_plot_url = None
    try:
        combined_data_for_beta = pd.concat([X_ref, target_sample_aligned], ignore_index=True)
        combined_data_for_beta = combined_data_for_beta.fillna(0).astype(float)

        if len(combined_data_for_beta) > 2: 
            dm = DistanceMatrix(squareform(pdist(combined_data_for_beta, metric='braycurtis')))
            pcoa_results = pcoa(dm)
            
            coords = pcoa_results.samples[['PC1', 'PC2']].values 

            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            
            age_months_ref_plot = reference_db['age_months'].values
            age_months_target_plot = predictions.get("age_months") 

            all_ages_for_plot = np.append(age_months_ref_plot, age_months_target_plot)
            
            scatter = ax.scatter(coords[:, 0], coords[:, 1], c=all_ages_for_plot, cmap='viridis', label='Amostras', alpha=0.7)
            
            ax.scatter(coords[-1, 0], coords[-1, 1], c='red', marker='X', s=150, label='Indivíduo Alvo', edgecolor='black', linewidth=1.5, zorder=10) 
            
            ax.set_xlabel('PCo1')
            ax.set_ylabel('PCo2')
            ax.set_title('PCoA da Composição da Microbiota por Idade (Distância Bray-Curtis)')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)

            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label('Idade em Meses')

            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            pcoa_plot_url = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig) 
        else:
            return render_template('results.html', error=f"Dados insuficientes para o cálculo do PCoA. Precisa de mais de 2 amostras combinadas para calcular as coordenadas.")


    except Exception as e:
        pcoa_plot_url = None
        print(f"Erro ao gerar o gráfico PCoA: {e}") 


    # Gráfico de Barras das 10 Bactérias Mais Abundantes no Indivíduo Alvo
    top_bacteria_plot_url = None
    top_3_bacteria_names = []
    try:
        target_bacteria_abundance_series = target_sample_aligned.iloc[0][species_columns] 
        
        target_bacteria_abundance_filtered = target_bacteria_abundance_series[target_bacteria_abundance_series > 0]
        
        top_10 = target_bacteria_abundance_filtered.nlargest(10) 
        
        if not top_10.empty:
            top_3_bacteria_names = top_10.head(3).index.tolist()

            if len(target_bacteria_abundance_filtered) > 10:
                others_sum = target_bacteria_abundance_filtered.drop(top_10.index, errors='ignore').sum()
                top_10 = pd.concat([top_10, pd.Series({'Outros': others_sum}, index=['Outros'])]) 
                
            fig_bar = Figure(figsize=(10, 6))
            ax_bar = fig_bar.add_subplot(111)
            
            labels = [label.replace('_', ' ').replace(' ', '\n') for label in top_10.index]
            
            ax_bar.bar(labels, top_10.values, color='skyblue')
            ax_bar.set_ylabel('Abundância Relativa')
            ax_bar.set_title('Bactérias Mais Abundantes no Indivíduo Alvo') 
            ax_bar.tick_params(axis='x', rotation=45) 
            fig_bar.tight_layout()

            buf_bar = io.BytesIO()
            fig_bar.savefig(buf_bar, format='png', bbox_inches='tight')
            buf_bar.seek(0)
            top_bacteria_plot_url = base64.b64encode(buf_bar.read()).decode('utf-8')
            plt.close(fig_bar)
        else:
            top_bacteria_plot_url = None 
            top_3_bacteria_names = [] 
            print("DEBUG: Nenhuma bactéria com abundância > 0 encontrada na amostra alvo para plotagem.")
            
    except Exception as e:
        top_bacteria_plot_url = None
        top_3_bacteria_names = [] 
        print(f"Erro ao gerar o gráfico de barras das bactérias: {e}")

    pubmed_search_query = ""
    pubmed_summary_text = ""
    try:
        pubmed_search_query = generate_pubmed_query_for_bacteria(top_3_bacteria_names)
        
        if pubmed_search_query:
            print(f"DEBUG: Consulta PubMed gerada: {pubmed_search_query}")
            article_summaries_raw = search_pubmed_and_get_summaries(pubmed_search_query, max_articles=5)
            
            if article_summaries_raw:
                pubmed_summary_text = summarize_articles_or_knowledge_with_gemini(article_summaries_raw, top_3_bacteria_names, num_articles_to_summarize=5)
            else:
                print("DEBUG: Sem resumos válidos do PubMed, caindo para resumo baseado em conhecimento prévio.")
                pubmed_summary_text = summarize_from_knowledge_gemini(top_3_bacteria_names)
        else:
            print("DEBUG: Consulta PubMed não gerada, caindo para resumo baseado em conhecimento prévio.")
            pubmed_summary_text = summarize_from_knowledge_gemini(top_3_bacteria_names)

    except Exception as e:
        pubmed_summary_text = f"Erro geral ao buscar ou resumir artigos PubMed: {e}. Caindo para resumo baseado em conhecimento prévio."
        print(f"DEBUG: Erro geral no fluxo PubMed/Resumo: {e}. Caindo para resumo baseado em conhecimento prévio.")
        pubmed_summary_text = summarize_from_knowledge_gemini(top_3_bacteria_names)


    display_predictions = predictions 


    return render_template('results.html', 
                           predictions=display_predictions, 
                           metrics=performance_metrics,
                           maz_value=maz_value,
                           alpha_diversity_value=alpha_diversity_value,
                           alpha_diversity_ref_mean=alpha_diversity_ref_mean, 
                           alpha_diversity_ref_std=alpha_diversity_ref_std,   
                           pcoa_plot_url=pcoa_plot_url,
                           top_bacteria_plot_url=top_bacteria_plot_url,
                           gemini_insight_text=pubmed_summary_text) 

if __name__ == '__main__':
    with app.app_context():
        configure_apis_global() 

    app.run(debug=True)