const LABEL_HATE = "hate"
const LABEL_OFFENSIVE = "offensive"
const LABEL_NONE = "none"
const BATCH_SIZE = 16

const url = "https://gkkhnyeypg.execute-api.ap-northeast-2.amazonaws.com/rocketll/hate-speech"

function isParagraph(node) {
    if (!node.hasChildNodes()) return false
    if (node.childElementCount == 0) return true

    let innerText = node.innerText.trim()
    const meaningLessFilterRegex = /\S\D/
    if (!meaningLessFilterRegex.test(innerText)) return false

    const oneWordFilterRegex = /\s/
    if (!oneWordFilterRegex.test(innerText)) return false

    var children = node.childNodes
    for (let i = 0; i < children.length; i++) {
        if ((children instanceof Element) && (children.tagName != "BR")) return false
    }

    return true
}

class HateBlocker {
    constructor(querySelectorString, level) {
        const allPossibleNodes = document.querySelectorAll(querySelectorString)
        var textNodes = new Array()
        var texts = new Array()
        
        for (let i = 0; i < allPossibleNodes.length; i++) {
            if (isParagraph(allPossibleNodes[i])) {
                textNodes.push(allPossibleNodes[i])
                texts.push(allPossibleNodes[i].innerText.trim())
            }
        }

        this.nodes = textNodes
        this.text = texts
        this.level = level
        this.querySelectorString = querySelectorString
    }

    getAddedContents(originalContents, level) {
        const allPossibleNodes = document.querySelectorAll(this.querySelectorString)
        var textNodes = new Array()
        var texts = new Array()
        
        for (let i = 0; i < allPossibleNodes.length; i++) {
            if (isParagraph(allPossibleNodes[i]) && !originalContents.includes(allPossibleNodes[i])) {
                textNodes.push(allPossibleNodes[i])
                texts.push(allPossibleNodes[i].innerText.trim())
            }
        }

        this.nodes = textNodes
        this.text = texts
        this.level = level
    }

    parseText() {
        // 문장 별로 나누기
        const seperatorRegex = /[\.|\n|\!|\?|;]/
        let rawSentences = this.text.split(seperatorRegex)

        // 의미없는 문장 지우기
        const meaningLessFilterRegex = /\S\D/
        const meaningfulSentences = rawSentences.filter((rawSentence) => meaningLessFilterRegex.test(rawSentence))

        const trimmedSentences = meaningfulSentences.map((meaningfulSentence) => meaningfulSentence.trim())

        // 한 단어짜리 문장 지우기
        const longSentences = trimmedSentences.filter((trimmedSentence) => /\s/.test(trimmedSentence))
        this.text = longSentences
    }

    generateBatch() {
        this.batch = new Array()
        for (let i = 0; i < this.text.length; i += BATCH_SIZE) {
            this.batch.push(this.text.slice(i, i + BATCH_SIZE))
        }
    }
    
    generateJson(batch) {
        var jsonObj = new Object()
        jsonObj.inputs = batch
        return JSON.stringify(jsonObj)
    }

    async apiResponse(originalContents) {
        if (this.nodes.length == 0) return
        var batchIndex = 0
        for (const batch of this.batch) {
            console.log("BATCH: " + batchIndex)
            const response = await fetch(url, {
                method: "POST",
                mode: "cors",
                headers: {
                    "Content-type": "application/json"
                },
                body: this.generateJson(batch)
            })
            await response.json().then(
                (res) => {
                    this.blockHate(res, batchIndex)
                }
            )
            batchIndex += 1
        }
        console.log("END OF BATCH LOOP")

        console.log("START NEW LOOP")
        var updatedContents = originalContents.concat(this.nodes.slice(0, this.nodes.length))
        this.getAddedContents(updatedContents, this.level)
        this.generateBatch()
        this.apiResponse(updatedContents)
    }

    blockHate(response, batchIndex) {
        const blockIndexes = new Array()

        for (let i = 0; i < response.length; i++) {
            if (this.level == 1) {
                if (response[i].label == LABEL_HATE) blockIndexes.push(i)
            } else if (this.level == 2) {
                if (response[i].label != LABEL_NONE) blockIndexes.push(i)
            }
        }
        
        for (const blockIndex of blockIndexes) {
            this.nodes[batchIndex * BATCH_SIZE + blockIndex].style.color = "transparent"
            console.log(this.nodes[batchIndex * BATCH_SIZE + blockIndex])
        }
    }
}

console.log("된다고 된다니까!")
const originalContents = []
const blocker = new HateBlocker('h1, h2, h3, h4, h5, p, li, td, div, caption, span, a, strong, article', 2)
blocker.generateBatch()
blocker.apiResponse(originalContents)
