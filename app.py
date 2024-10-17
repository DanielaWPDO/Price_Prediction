import gradio as gr
import pickle
import pandas as pd
from babel.numbers import format_currency

def predict(code_postal,adresse_voie,surface_terrain,type_local,surface_reelle_bati,nombre_pieces_principales):
    with open('encoders.pkl', 'rb') as f:
        encoder = pickle.load(f)
    pickle_model=pickle.load(open('model.pkl','rb'))
    
    request={
    "data":{
        "code_postal":code_postal,
        "adresse_nom_voie":adresse_voie,
        "surface_terrain":surface_terrain,
        "type_local":type_local,
        "surface_reelle_bati":surface_reelle_bati,
        "nombre_pieces_principales":nombre_pieces_principales  
        }
    }
    # Encode 'Voie' column
    data = pd.json_normalize(request['data'])
    data['adresse_nom_voie'] = encoder['adresse_nom_voie'].transform(data[['adresse_nom_voie']])
    data['type_local']=encoder['type_local'].transform(data[['type_local']])
    prediction= pickle_model.predict(data)
    return str(format_currency(int(prediction[0]), 'EUR', locale='fr_FR'))

with gr.Blocks() as demo:
    gr.Markdown("<h1>Put your dream accommodation's information and we'll estimate it!!!</h1>")
    with gr.Row():
        i1 = gr.Number(label='Please provide your postal code')
        i2 = gr.Text(label='Street address: ')
        i3= gr.Number(label='Surface area: ')
        i4= gr.Dropdown(
                    ["Maison", "Appartement",'Dépendance'], label="Room type")
        i5= gr.Number(label='Actual surface area of the building : ')
        i6= gr.Slider(0,10,step= 1, label='Number of principal rooms: ', info="Choose between 0 and 10")
    with gr.Row():
        b = gr.Button(value='Estimate')
        output=gr.Textbox(label='Estimate price of premises')
        b.click(predict, inputs=[i1, i2, i3, i4,i5,i6], outputs=output)
if __name__ == "__main__":
    demo.launch(debug=True)