require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./load-csv');

const LinearRegression = require('./linear-regression');
const plot = require('node-remote-plot');

let  { features, labels, testFeatures, testLabels} = loadCSV('./UpdatedData2.csv',{
    shuffle: true,
    splitTest: 50,
    dataColumns: ['Latitude','Longitude'],
    labelColumns: ['Pricing']
});

// console.log(features);
console.log(features);

// console.log(features, labels);

const regression = new LinearRegression(features, labels, {
    learningRate: 0.1,
    iteration: 3,
    batchSize: 10
});

// regression.features.print();

regression.train();

// console.log('Updates M is:', regression.m, 'Updated B is:', regression.b)
// console.log("Updated M is:", regression.weights.get(1,0),"Updated B is: ",regression.weights.get(0,0));

const r2 = regression.test(testFeatures, testLabels);

// console.log('MSE Histroy: ', regression.mseHistroy);

plot({
    x: regression.mseHistory.reverse(),
    xLabel: "Iteration Number",
    yLabel: 'Mean Squared Error'
});
console.log("R2 is: ", r2);

// let v1 = regression.predict(
// //     [
// //         [120, 2, 380]
// //     ]

// // ).sum();

// let v1 = tf.variable();

// console.log(v1);

// let v1 = regression.predict([[22.3, 73.0, 2]]).flatten().buffer().get(0);

// // let t2 = tf.buffer(v1).toTensor().print();

// // let v2 = tf.scalar(1, 'float32');
// // let v2 = tf.cast(v1, 'float32').print();

// // let v2 = v1.toFloat();

// // console.log("Happy Birthday " + v2);

// //  console.log(v3);

// // const v2 = t2.get(2);
// // console.log(v2);

// ------------------------ Main Content ----------------------------/

let v1 = regression.predict([[22.3, 73.0]]).flatten().buffer().get(0);
let v2 = v1.toFixed(2);

console.log('Pricing :', v2);






// regression.predict([[120,2,370]]).print();
