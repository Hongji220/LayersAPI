const model = tf.sequential();

let learningRate = 0.1;


// Creating the single hidden layer in this model, specifying the inputs.
// dense is a "fully connected later"
const hidden = tf.layers.dense({
	units: 4,   // number of nodes
	inputShape: [2],  //input shape 
	activation: 'sigmoid'
});

// Adding it to the model:
model.add(hidden);

//Creating the output layers:

const output = tf.layers.dense({
	units: 3,
	// the input shape is inferred from the previous layer,
	// so it does not have to be specified.
	activation: 'sigmoid'
});

// Adding it to the model:
model.add(output);

//Creating the Stochastic gradient descent optimizer with LR.

const sgdOptimizer = tf.train.sgd(learningRate);

// Calling the tf.model compiler to compile with those configurations:

model.compile({
	optimizer: sgdOptimizer,
	loss: 'meanSquaredError'  // or loss: tf.losses.meanSquaredError
});

const xs = tf.tensor2d([
	[0.1, 0.2],
	[0.2, 0.4],
	[0.3, 0.5]
])

const ys = tf.tensor2d([
	[0.2, 0.2 , 0.2],
	[0.2, 0.2 , 0.2],
	[0.2, 0.2 , 0.2]
])


// We can do this, but it can only  be run once and cannot be reiterated (it is async).
/*model.fit(xs,ys).then((response) => console.log(response));*/

async function train() {
	for (let i = 0; i < 100 ; i++){
		const history = await model.fit(xs,ys, {
			epochs: 10,
			shuffle:true
		}).then((response)=> {
			console.log(response.history.loss);
			}
		)};
	}
	




train().then(() => { 
	console.log("Training Complete.");
	let outputs = model.predict(xs);
	outputs.print();
});
















