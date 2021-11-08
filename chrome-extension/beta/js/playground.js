const url =
  "https://gkkhnyeypg.execute-api.ap-northeast-2.amazonaws.com/rocketll/hate-speech";
const BATCH_SIZE = 16;
var input = document.getElementById("input");
var output = document.getElementById("output");
var text;
// var button = document.getElementById("submit")

function submit() {
  if (input) {
    text = input.value;
    console.log("starts..");
    const blocker = new HateBlocker(text, 1);
    blocker.parseText();
    blocker.generateBatch();
    responses = blocker.apiResponse();
    console.log("response printing");
    console.log(responses);
  }
}

class HateBlocker {
  constructor(texts, level) {
    this.text = texts;
    this.level = level;
  }

  parseText() {
    // 문장 별로 나누기
    const seperatorRegex = /[\.|\n|\!|\?|;]/;
    let rawSentences = this.text.split(seperatorRegex);

    // 의미없는 문장 지우기
    const meaningLessFilterRegex = /\S\D/;
    const meaningfulSentences = rawSentences.filter((rawSentence) =>
      meaningLessFilterRegex.test(rawSentence)
    );

    const trimmedSentences = meaningfulSentences.map((meaningfulSentence) =>
      meaningfulSentence.trim()
    );

    // 한 단어짜리 문장 지우기
    const longSentences = trimmedSentences.filter((trimmedSentence) =>
      /\s/.test(trimmedSentence)
    );
    this.text = longSentences;
  }

  generateBatch() {
    this.batch = new Array();
    for (let i = 0; i < this.text.length; i += BATCH_SIZE) {
      this.batch.push(this.text.slice(i, i + BATCH_SIZE));
    }
  }

  generateJson(batch) {
    var jsonObj = new Object();
    jsonObj.inputs = batch;
    return JSON.stringify(jsonObj);
  }

  async apiResponse() {
    for (const batch of this.batch) {
      console.log(batch.length);
      const response = await fetch(url, {
        method: "POST",
        mode: "cors",
        headers: {
          "Content-type": "application/json",
        },
        body: this.generateJson(batch),
      });
      await response.json().then((res) => {
        console.log("BATCH DONE");
        console.log(res);
        return res;
      });
    }
  }
}
