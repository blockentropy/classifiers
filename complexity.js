import { pipeline, env, cos_sim, AutoTokenizer, AutoModelForSequenceClassification } from '@xenova/transformers';
//const env = require('@xenova/transformers').env;
// Specify a custom location for models (defaults to '/models/').
env.localModelPath = './models/';

// Disable the loading of remote models from the Hugging Face Hub:
env.allowRemoteModels = false;

// Set location of .wasm files. Defaults to use a CDN.
env.backends.onnx.wasm.wasmPaths = './models/';
console.log(env.backends.onnx);

const MAX_TOKENS = 475;

let classifier = await pipeline('zero-shot-classification', './distilbert-base-uncased-mnli');
let sentiment = await pipeline('text-classification', './distilbert-base-uncased-finetuned-sst-2-english');
let extractor = await pipeline('feature-extraction', './bge-base-en-v1.5');

let llamatokenizer = await AutoTokenizer.from_pretrained('llama2.c-stories15M');
let tokenizer = await AutoTokenizer.from_pretrained('./finetuned');
let model = await AutoModelForSequenceClassification.from_pretrained('./finetuned');
let modelcode = await AutoModelForSequenceClassification.from_pretrained('./finetunedcode');
let modelreason = await AutoModelForSequenceClassification.from_pretrained('./finetunedreason');
let modelshort = await AutoModelForSequenceClassification.from_pretrained('./finetunedshort');
let modellong = await AutoModelForSequenceClassification.from_pretrained('./finetunedlong');
let modelroleplay = await AutoModelForSequenceClassification.from_pretrained('./finetunedroleplay');
let modelcot = await AutoModelForSequenceClassification.from_pretrained('./finetunedcot');

export async function tokenText(text) {
  let { input_ids, attention_mask } = await tokenizer(text);

  // Calculate the token count before slicing
  const tokenCount = input_ids.data.length;

  // Only take the last 512 tokens if the count exceeds the MAX_TOKENS
  if (tokenCount > MAX_TOKENS) {
    input_ids.data = input_ids.data.slice(-MAX_TOKENS);
    attention_mask.data = attention_mask.data.slice(-MAX_TOKENS);

    // Update the dims property to reflect the new shape
    input_ids.dims = [1, MAX_TOKENS];
    attention_mask.dims = [1, MAX_TOKENS];
  }

  return { input_ids, attention_mask, tokenCount };
}

export async function llamaToken(text) {
  let { input_ids, attention_mask } = await llamatokenizer(text);

  // Calculate the token count before slicing
  const tokenCount = input_ids.data.length;

  return tokenCount;
}

export async function classifyText(text, classes = ['politics', 'not politics']) {
  let { input_ids, attention_mask} = await tokenizer(text);
  // Only take the last 512 tokens
  if (input_ids.dims[1] > MAX_TOKENS) {
    input_ids.data = input_ids.data.slice(-MAX_TOKENS);
    input_ids.dims = [1, MAX_TOKENS];
  }
  const tokenIdsArray = Array.from(input_ids.data).map(BigInt => Number(BigInt));
    
  // Convert token IDs back to string using the tokenizer's decode function
  const truncatedText = tokenizer.decode(tokenIdsArray);

  return await classifier(truncatedText, classes);
}

export async function classifySentiment(text) {
  let { input_ids, attention_mask} = await tokenizer(text);
  // Only take the last 512 tokens
  if (input_ids.dims[1] > MAX_TOKENS) {
    input_ids.data = input_ids.data.slice(-MAX_TOKENS);
    input_ids.dims = [1, MAX_TOKENS];
  }
  const tokenIdsArray = Array.from(input_ids.data).map(BigInt => Number(BigInt));
    
  // Convert token IDs back to string using the tokenizer's decode function
  const truncatedText = tokenizer.decode(tokenIdsArray);

  return await sentiment(truncatedText);
}

export async function classifyComplexity(input_ids, attention_mask) {
return await model({ input_ids, attention_mask });

}

export async function classifyCode(input_ids, attention_mask) {
return await modelcode({ input_ids, attention_mask });
}

export async function classifyReason(input_ids, attention_mask) {
return await modelreason({ input_ids, attention_mask });
}

export async function classifyShort(input_ids, attention_mask) {
  return await modelshort({ input_ids, attention_mask });
}

export async function classifyLong(input_ids, attention_mask) {
  return await modellong({ input_ids, attention_mask });
}

export async function classifyRoleplay(input_ids, attention_mask) {
  return await modelroleplay({ input_ids, attention_mask });
}

export async function classifyCot(input_ids, attention_mask) {
  return await modelcot({ input_ids, attention_mask });
}

export async function createEmbedding(text) {
  return await extractor(text, { pooling: 'mean', normalize: true });
}

export function cosSimilarity(embedding1, embedding2) {
  return cos_sim(embedding1, embedding2);
}