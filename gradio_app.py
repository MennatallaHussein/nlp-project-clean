import gradio as gr
from Theme_classifier import ThemeClassifier
from character_network import NamedEntityRecognizer, CharacterNetworkGenerator
from text_classification import JutsuClassifier
import os
from dotenv import load_dotenv
from character_chatbot import CharacterChatBot

load_dotenv()

def get_themes(theme_list_str,subtitles_path,save_path):
    theme_list=theme_list_str.split(',')
    theme_classifier=ThemeClassifier(theme_list)
    output_df=theme_classifier.get_theme(subtitles_path,save_path)

    theme_list = [theme for theme in theme_list if theme !='dialogue']
    output_df=output_df[theme_list]

    output_df=output_df[theme_list].sum().reset_index()
    output_df.columns=['theme','score']


    output_chart= gr.BarPlot(
        output_df,
        x='theme' ,
        y='score',
        title="Series Themes",
        tooltip=['theme', 'score'],
        width=500,
        height=260
        
    )


    return output_chart



def get_character_network(subtitles_path,ner_path):
    ner= NamedEntityRecognizer()
    ner_df= ner.get_ners(subtitles_path,ner_path)

    character_network_generator=CharacterNetworkGenerator()
    relationship_df=character_network_generator.generate_character_network(ner_df)

    html=character_network_generator.draw_network_graph(relationship_df)


    return html


def classify_text(text_classification_model,text_classification_data,text_to_classify):
    
    jutsu_classifier=JutsuClassifier(model_path=text_classification_model,
                                     data_path=text_classification_data,
                                     huggingface_token=os.getenv('huggingface_token'))
    output=jutsu_classifier.classify_jutsu(text_to_classify)
    
    
    return output[0]





def chat_with_character_chatbot(message, history):
    character_Chatbot=CharacterChatBot("mennaGHANAM/Naruto-Llama-3-8B",
                                        huggingface_token=os.getenv('huggingface_token'),
                                        )
    output=character_Chatbot.chat(message,history)
   
    return output.strip()









def main():
    with gr.Blocks() as iface:

        # Theme Classification Section
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Theme Classification (Zero shot Classifiers)</h1>")
        
                with gr.Row():
                    with gr.Column():
                        plot=gr.BarPlot()
                    with gr.Column():
                        theme_list=gr.Textbox(label="Themes")
                        subtitles_path=gr.Textbox(label="Subtitles or script path")
                        save_path=gr.Textbox(label="Save path")
                        get_themes_button= gr.Button("Get Themes")
                        get_themes_button.click(get_themes, inputs=[theme_list,subtitles_path,save_path], outputs=plot)
  


       # char network sec
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Character Network (NERs and Graphs)</h1>")
                with gr.Row():
                    with gr.Column():
                        network_html=gr.HTML()
                    with gr.Column():
                        subtitles_path=gr.Textbox(label="Subtitles or script path")
                        ner_path=gr.Textbox(label="NERs save path")
                        get_network_graph_button= gr.Button("Get Character Network")
                        get_network_graph_button.click(get_character_network, inputs=[subtitles_path,ner_path], outputs=[network_html])




       # Text classification with lllms
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Text Classification with LLMs</h1>")
                with gr.Row():
                    with gr.Column():
                        text_classification_output=gr.Textbox(label="Text Classification Output")
                    with gr.Column():

                        text_classification_model=gr.Textbox(label="Model path")
                        text_classification_data=gr.Textbox(label="Data path")
                        text_to_classify=gr.Textbox(label="Text input")

                        classify_text_button= gr.Button("Classify text")
                        classify_text_button.click(classify_text, inputs=[text_classification_model,text_classification_data,text_to_classify], outputs=[text_classification_output])



       # character chatbot sec
        with gr.Row():
            with gr.Column():
                gr.HTML("<h1>Chsrscter Chatbot</h1>")
                gr.ChatInterface(chat_with_character_chatbot)







    iface.launch(share=True)





if __name__ == '__main__':
    main()
