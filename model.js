const tf = require('@tensorflow/tfjs-node')
const props = require("./data/web_prop_data.json")



async function loadModel() {
    console.log('Loading Model 1...')
    model = await tf.loadLayersModel("file://model/model.json", false);
    console.log('Model Loaded Successfull')
    // model.summary()
}

const prop_arr = tf.range(0, props.length)
const prop_len = props.length


exports.recommend = async function recommend(userId) {
    let user = tf.fill([prop_len], Number(userId))
    let prop_in_js_array = prop_arr.arraySync()
    await loadModel()
    console.log(`Recommending for User: ${userId}`)
    pred_tensor = await model.predict([prop_arr, user]).reshape([10000])
    pred = pred_tensor.arraySync()
    
    let recommendations = []
    for (let i = 0; i < 10; i++) {
        max = pred_tensor.argMax().arraySync()
        recommendations.push(props[max]) //Push prop with highest prediction probability
        pred.splice(max, 1)    //drop from array
        pred_tensor = tf.tensor(pred) //create a new tensor
    }
    
    return recommendations


}