// var a = [{ "tokens": ["In", "order", "to", "dynamically", "learn", "filters", "and", "features", "we", "look", "to", "Convolutional", "Neural", "Networks", "(", "CNNs", ")", "which", "have", "shown", "their", "dominance", "in", "computer", "vision"], "intent_prediction": { "AI2020": 1 }, "slot_prediction": { "AI2020": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "DEF", "DEF", "DEF", "O", "TERM", "O", "O", "O", "O", "O", "O", "O", "O", "O"] } }, { "tokens": ["Convolutional", "Neural", "Networks", ",", "CNNs", ",", "have", "shown", "their", "dominance", "in", "computer", "vision", "(", "CV", ")"], "intent_prediction": { "AI2020": 1 }, "slot_prediction": { "AI2020": ["DEF", "DEF", "DEF", "O", "TERM", "O", "O", "O", "O", "O", "O", "DEF", "DEF", "O", "TERM", "O"] } }, { "tokens": ["The", "input", "to", "the", "matrix", "[[M]]", "is", "a", "vector", "[[v]]", "of", "specified", "lengths", "."], "intent_prediction": { "AI2020": 0 }, "slot_prediction": { "AI2020": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"] } }]

function prettyPrintArray(json) {
    if (typeof json === 'string') {
        json = JSON.parse(json);
    }
    output = JSON.stringify(json, function (k, v) {
        if (v instanceof Array)
            return JSON.stringify(v);
        return v;
    }, 2).replace(/\\/g, '')
        .replace(/\"\[/g, '[')
        .replace(/\]\"/g, ']')
        .replace(/\"\{/g, '{')
        .replace(/\}\"/g, '}');

    return output;
}

$('#predict').click(() => {
    let text = $('#text-area').val();
    console.log(text);
    $.ajax({
        url: 'http://192.168.1.195:5000/get_prediction',
        crossDomain: true,
        dataType: 'json',
        data: { text: text },
        success: (d) => {
            console.log(d);
            d = d.results;
            output = '[\n';
            for (var i = 0; i < d.length; i++) {
                output += prettyPrintArray(d[i]) + ',\n';
            }
            output += '\n]';
            $('#output').html(output)


        }
    });
})
