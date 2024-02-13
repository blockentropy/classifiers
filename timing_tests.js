import { createEmbedding, cosSimilarity, classifyText, classifyCode, classifyCot, classifyShort, classifyReason, classifyLong, tokenText } from './complexity.js';

let texts = [
    'Coding',
    'Not Coding',
];

const numruns = 20;

////////////////////////////
// Embedding Approach /////
///////////////////////////

// Start measuring time
let startTime = process.hrtime.bigint();

// Compute sentence embeddings
let embeddings = await createEmbedding(texts);

// Prepend recommended query instruction for retrieval.
let query = 'write in python';

//let query = query_prefix + 'What is the difference between central diabetes insipidus in children and adolescents versus adults, and how do treatment options differ based on the underlying cause?';
let query_embeddings = await createEmbedding(query);

// Sort by cosine similarity score
let scores = embeddings.tolist().map(
    (embedding, i) => ({
        id: i,
        score: cosSimilarity(query_embeddings.data, embedding),
        text: texts[i],
    })
).sort((a, b) => b.score - a.score);
//console.log(scores);

// End measuring time
let endTime = process.hrtime.bigint();

//console.log(scores);
console.log(`WARMUP Execution time: ${(endTime - startTime) / BigInt(1000000)} milliseconds`);

// Repeat 10 times and calculate average execution time
let totalExecutionTime = BigInt(0);
for (let i = 0; i < numruns; i++) {
    // Start measuring time
    startTime = process.hrtime.bigint();

    // Compute sentence embeddings
    embeddings = await createEmbedding(texts);
    query_embeddings = await createEmbedding(query);

    // Sort by cosine similarity score
    scores = embeddings.tolist().map(
        (embedding, i) => ({
            id: i,
            score: cosSimilarity(query_embeddings.data, embedding),
            text: texts[i],
        })
    ).sort((a, b) => b.score - a.score);

    // End measuring time
    endTime = process.hrtime.bigint();
    totalExecutionTime += endTime - startTime;
}
let averageExecutionTime = totalExecutionTime / BigInt(numruns);
console.log(`Average Execution time EMBEDDINGS over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);

/////////////////////
// ZERO SHOT APPROACH
/////////////////////

// Start measuring time
startTime = process.hrtime.bigint();

// Compute sentence embeddings
scores = await classifyText(query, texts);
//console.log(scores);

// End measuring time
endTime = process.hrtime.bigint();

//console.log(scores);
console.log(`WARMUP Execution time: ${(endTime - startTime) / BigInt(1000000)} milliseconds`);

// Repeat 10 times and calculate average execution time
totalExecutionTime = BigInt(0);
for (let i = 0; i < numruns; i++) {
    // Start measuring time
    startTime = process.hrtime.bigint();
    scores = await classifyText(query, texts);
    endTime = process.hrtime.bigint();
    totalExecutionTime += endTime - startTime;
}
averageExecutionTime = totalExecutionTime / BigInt(numruns);
console.log(`Average Execution time ZERO SHOT over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);


/////////////////////////////////
// PRETRAINED CLASSIFIER APPROACH
/////////////////////////////////

// Start measuring time
startTime = process.hrtime.bigint();
let input_ids, attention_mask, tokenCount, logits; // Declare the variables here
// Compute sentence embeddings
({ input_ids, attention_mask, tokenCount } = await tokenText(query));
({ logits } = await classifyCode(input_ids, attention_mask));
({ logits } = await classifyCot(input_ids, attention_mask)); // Assign values using destructuring
({ logits } = await classifyShort(input_ids, attention_mask)); // Assign values using destructuring
({ logits } = await classifyLong(input_ids, attention_mask)); // Assign values using destructuring
({ logits } = await classifyReason(input_ids, attention_mask)); // Assign values using destructuring
//console.log(logits);

// End measuring time
endTime = process.hrtime.bigint();

console.log(`WARMUP Execution time: ${(endTime - startTime) / BigInt(1000000)} milliseconds`);

// Repeat 10 times and calculate average execution time
totalExecutionTime = BigInt(0);

for (let i = 0; i < numruns; i++) {
    // Start measuring time
    startTime = process.hrtime.bigint();
    ({ input_ids, attention_mask, tokenCount } = await tokenText(query)); // Assign values using destructuring
    ({ logits } = await classifyCode(input_ids, attention_mask)); // Assign values using destructuring
    endTime = process.hrtime.bigint();
    totalExecutionTime += endTime - startTime;
}
averageExecutionTime = totalExecutionTime / BigInt(numruns);
console.log(`Average Execution time CLASSIFER over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);
console.log(tokenCount);


///// INCREASE Size of Query////


for (let i = 0; i < 2; i++) {
    query = query + query;

    totalExecutionTime = BigInt(0);
    for (let i = 0; i < numruns; i++) {
        // Start measuring time
        startTime = process.hrtime.bigint();

        // Compute sentence embeddings
        embeddings = await createEmbedding(texts);
        query_embeddings = await createEmbedding(query);

        // Sort by cosine similarity score
        scores = embeddings.tolist().map(
            (embedding, i) => ({
                id: i,
                score: cosSimilarity(query_embeddings.data, embedding),
                text: texts[i],
            })
        ).sort((a, b) => b.score - a.score);

        // End measuring time
        endTime = process.hrtime.bigint();
        totalExecutionTime += endTime - startTime;
    }
    averageExecutionTime = totalExecutionTime / BigInt(numruns);
    console.log(`Average Execution time EMBEDDINGS over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);

    totalExecutionTime = BigInt(0);
    for (let i = 0; i < numruns; i++) {
        // Start measuring time
        startTime = process.hrtime.bigint();
        scores = await classifyText(query, texts);
        endTime = process.hrtime.bigint();
        totalExecutionTime += endTime - startTime;
    }
    averageExecutionTime = totalExecutionTime / BigInt(numruns);
    console.log(`Average Execution time ZERO SHOT over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);

    // Repeat 10 times and calculate average execution time
    totalExecutionTime = BigInt(0);

    for (let i = 0; i < numruns; i++) {
        // Start measuring time
        startTime = process.hrtime.bigint();
        ({ input_ids, attention_mask, tokenCount } = await tokenText(query)); // Assign values using destructuring
        ({ logits } = await classifyCode(input_ids, attention_mask)); // Assign values using destructuring
        endTime = process.hrtime.bigint();
        totalExecutionTime += endTime - startTime;
    }
    averageExecutionTime = totalExecutionTime / BigInt(numruns);
    console.log(`Average Execution time CLASSIFER over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);
    console.log(`Token Count: ${tokenCount}`);
}


/////
//////////////////////
/// INCREASE NUMBER OF CLASSIFIERS
//////////////////////
/*texts = [
    'Coding',
    'Not Coding',
    'Chain of Thought',
    'Not Chain of Thought',
];*/
/*texts = [
    'Coding',
    'Not Coding',
    'Chain of Thought',
    'Not Chain of Thought',
    'Long',
    'Not Long',
];*/
/*texts = [
    'Coding',
    'Not Coding',
    'Chain of Thought',
    'Not Chain of Thought',
    'Long',
    'Not Long',
    'Short',
    'Not Short',
];*/


////// INCREASE SIZE OF EMBEDDINGS/////

texts = [
    'Coding',
    'Not Coding',
    'Chain of Thought',
    'Not Chain of Thought',
    'Long',
    'Not Long',
    'Short',
    'Not Short',
    'Reasoning',
    'Not Reasoning',
];
/*texts = [
    'Coding Coding Coding Coding ',
    'Not Coding Not Coding Not Coding Not Coding',
    'Chain of Thought Chain of Thought Chain of Thought Chain of Thought',
    'Not Chain of Thought Not Chain of Thought Not Chain of Thought Not Chain of Thought',
    'Long Long Long Long',
    'Not Long Not Long Not Long Not Long',
    'Short Short Short Short',
    'Not Short Not Short Not Short Not Short',
    'Reasoning Reasoning Reasoning Reasoning',
    'Not Reasoning Not Reasoning Not Reasoning Not Reasoning',
];*/ // 76 words
/*texts = [
    'Coding Coding Coding Coding Coding Coding Coding Coding ',
    'Not Coding Not Coding Not Coding Not Coding Not Coding Not Coding Not Coding Not Coding',
    'Chain of Thought Chain of Thought Chain of Thought Chain of Thought Chain of Thought Chain of Thought Chain of Thought Chain of Thought',
    'Not Chain of Thought Not Chain of Thought Not Chain of Thought Not Chain of Thought Not Chain of Thought Not Chain of Thought Not Chain of Thought Not Chain of Thought',
    'Long Long Long Long Long Long Long Long',
    'Not Long Not Long Not Long Not Long Not Long Not Long Not Long Not Long',
    'Short Short Short Short Short Short Short Short',
    'Not Short Not Short Not Short Not Short Not Short Not Short Not Short Not Short',
    'Reasoning Reasoning Reasoning Reasoning Reasoning Reasoning Reasoning Reasoning',
    'Not Reasoning Not Reasoning Not Reasoning Not Reasoning Not Reasoning Not Reasoning Not Reasoning Not Reasoning',
]; */// 152 words

// Function to increase the size of each text in the array by repeating it with spaces
function increaseTextSizeAndCountWords(textArray) {
    const modifiedTexts = textArray.map(text => `${text} ${text}`);
    
    // Calculating the total word count
    const totalWordCount = modifiedTexts.reduce((count, text) => count + text.split(' ').length, 0);
    console.log(`Total word count: ${totalWordCount}`);
    return modifiedTexts;
}



for (let i = 0; i < 20; i++) {
    //query = "what is python";
    texts = increaseTextSizeAndCountWords(texts);

    totalExecutionTime = BigInt(0);
    for (let i = 0; i < numruns; i++) {
        // Start measuring time
        startTime = process.hrtime.bigint();

        // Compute sentence embeddings
        embeddings = await createEmbedding(texts);
        query_embeddings = await createEmbedding(query);

        // Sort by cosine similarity score
        scores = embeddings.tolist().map(
            (embedding, i) => ({
                id: i,
                score: cosSimilarity(query_embeddings.data, embedding),
                text: texts[i],
            })
        ).sort((a, b) => b.score - a.score);

        // End measuring time
        endTime = process.hrtime.bigint();
        totalExecutionTime += endTime - startTime;
    }
    averageExecutionTime = totalExecutionTime / BigInt(numruns);
    console.log(`Average Execution time multi EMBEDDINGS over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);

    totalExecutionTime = BigInt(0);
    for (let i = 0; i < numruns; i++) {
        // Start measuring time
        startTime = process.hrtime.bigint();
        scores = await classifyText(query, texts);
        endTime = process.hrtime.bigint();
        totalExecutionTime += endTime - startTime;
    }
    averageExecutionTime = totalExecutionTime / BigInt(numruns);
    console.log(`Average Execution time multi ZERO SHOT over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);

    // Repeat 10 times and calculate average execution time
    totalExecutionTime = BigInt(0);
    ({ input_ids, attention_mask, tokenCount } = await tokenText(query)); // Assign values using destructuring
    for (let i = 0; i < numruns; i++) {
        // Start measuring time
        startTime = process.hrtime.bigint();
        ({ logits } = await classifyCode(input_ids, attention_mask)); // Assign values using destructuring
        ({ logits } = await classifyCot(input_ids, attention_mask)); // Assign values using destructuring
        ({ logits } = await classifyLong(input_ids, attention_mask)); // Assign values using destructuring
        ({ logits } = await classifyShort(input_ids, attention_mask)); // Assign values using destructuring
        ({ logits } = await classifyReason(input_ids, attention_mask)); // Assign values using destructuring
        endTime = process.hrtime.bigint();
        totalExecutionTime += endTime - startTime;
    }
    averageExecutionTime = totalExecutionTime / BigInt(numruns);
    console.log(`Average Execution time multi CLASSIFER over ${numruns} runs: ${averageExecutionTime / BigInt(1000000)} milliseconds`);
    console.log(`Token Count: ${tokenCount}`);
}
