const tf = require('@tensorflow/tfjs');
const _ = require('lodash');

class LinearRegression{

    constructor(features, labels, options){

        // this.features = tf.tensor(features);
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = []; 
        // this.bHistory = []; // all the histroy of vakues of b

        //ones
        // this.features = tf.ones([this.features.shape[0], 1]).concat(this.features,1);;

        // this.options = options;
        this.options = Object.assign({learningRate: 0.1, iteration: 1000}, options);

        // this.m = 0;
        // this.b =0;
        
        this.weights = tf.zeros([this.features.shape[1],1]);
    }

    gradientDescent(features, labels){

       const currentGuess =  features.matMul(this.weights);

       const differences = currentGuess.sub(labels);

       const slopes = features
        .transpose()
        .matMul(differences)
        .div(features.shape[0]);

      this.weights = this.weights.sub(slopes.mul(this.options.learningRate));

    }

    // gradientDescent(){

    //     const currentGuessesForMPG = this.features.map(row => {
    //         return this.m * row[0] + this.b;
    //     });

    //     const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {

    //         return guess - this.labels[i][0];

    //     })) * 2  / this.features.length;

    //     const mSlope = _.sum(currentGuessesForMPG.map((guess, i)=> {
    //         return -1 * this.features[i][0] * (this.labels[i][0] - guess);
    //     })) * 2 / this.features.length;

    //     this.m = this.m - mSlope * this.options.learningRate;
    //     this.b = this.b - bSlope * this.options.learningRate;
    // }

    train(){
      const batchQuantity = Math.floor(this.features.shape[0] / this.options.batchSize);
        
      
      
      for (let i =0;i< this.options.iteration;i++)
        {
            for(let j=0;j< batchQuantity;j++)
            {
                const startIndex = j * this.options.batchSize;
                const { batchSize } = this.options;
                const featureSlice = this.features.slice([startIndex,0],[batchSize, -1]);

                const labelSLice = this.labels.slice(
                    [startIndex,0],
                    [batchSize, -1]
                );

                this.gradientDescent(featureSlice, labelSLice);
            }
            // console.log(this.options.learningRate);
            // this.gradientDescent();
            this.recordMSE();
            this.updateLearningRate();
        }

    }

    predict(observations){

        return this.processFeatures(observations)
            .matMul(this.weights);

    }

    test(testFeatures, testLabels){

        // testFeatures = tf.tensor(testFeatures);
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        // testFeatures = tf.ones([testFeatures.shape[0],1]).concat(testFeatures,1);

        

        const predictions = testFeatures.matMul(this.weights);

        // predictions.print();

       const res = testLabels
        .sub(predictions)
        .pow(2)
        .sum()
        .get();

        const tot = testLabels
                    .sub(testLabels.mean())
                    .pow(2)
                    .sum()
                    .get();

        return 1-res/tot;
        

        }   

        processFeatures(features)
        {
            features = tf.tensor(features);


            if(this.mean && this.variance)
            {
                features = features.sub(this.mean).div(this.variance.pow(0.5));
            }
            else{
                features = this.standardize(features);
            }

            features = tf.ones([features.shape[0],1]).concat(features,1);
            
            return features;
        }

        standardize(features)
        {
            const { mean, variance} = tf.moments(features, 0);

            this.mean = mean;
            this.variance = variance;

            return features.sub(mean).div(variance.pow(0.5));
        }

        //calculate and record Mean Squared Error MSE

        recordMSE(){
           const mse = this.features
                .matMul(this.weights)
                .sub(this.labels)
                .pow(2)
                .sum()
                .div(this.features.shape[0])
                .get();

                this.mseHistory.unshift(mse);
        }

        //last 2 MSE Values and Updating Learning Rate
        updateLearningRate()
        {
            if(this.mseHistory.length < 2)
            {
                return;
            }

            // const lastValue = this.mseHistroy[this.mseHistroy.length -1];
            // const secondLast = this.mseHistroy[this.mseHistroy.length -2];

            if(this.mseHistory[0] > this.mseHistory[1])
            {
                this.options.learningRate = this.options.learningRate/2;
            }
            else{
                this.options.learningRate *= 1.05;
            }
        }

}

module.exports = LinearRegression;