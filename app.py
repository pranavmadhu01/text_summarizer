#import statements
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from flask import Flask, render_template, request

# necessary variables
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
the_encoder = model.get_encoder()
the_decoder = model.get_decoder()
last_linear_layer = model.lm_head

# initializing app
app = Flask("Text Summarizer")


#opening of the app
@app.route("/")
def home():
    return render_template("input.html")
# post request code
@app.route("/process", methods=["POST"])
def prediction():
    ARTICLE_TO_SUMMARIZE = request.form['projectFilepath']
    tokenized_input = tokenizer(
        [ARTICLE_TO_SUMMARIZE], max_length=1024, truncation=True, return_tensors='pt')
    input_representation = the_encoder(input_ids=tokenized_input.input_ids,
                                       attention_mask=tokenized_input.attention_mask)
    start_token = torch.tensor([tokenizer.bos_token_id]).unsqueeze(0)
    mask_ids = torch.tensor([1]).unsqueeze(0)
    decoder_output = the_decoder(input_ids=start_token,
                                 attention_mask=mask_ids,
                                 encoder_hidden_states=input_representation[0],
                                 encoder_attention_mask=tokenized_input.attention_mask)

    decoder_output = decoder_output.last_hidden_state
    summary_ids = model.generate(
        tokenized_input['input_ids'], max_length=100, early_stopping=True)
    return render_template('output.html', myoutput=[tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])


if __name__ == "__main__":
    app.debug = True
    app.run()
