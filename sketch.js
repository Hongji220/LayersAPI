window.onload(drawGraph())

function drawGraph(){


const model = tf.sequential();

let learningRate = 0.8;
let lossovertime = [];
	

// We can do this, but it can only  be run once and cannot be reiterated (it is async).
/*model.fit(xs,ys).then((response) => console.log(response));*/

async function train() {
	for (let i = 0; i < 50 ; i++){
		const history = await model.fit(xs,ys, {
			shuffle:true
		}).then((response)=> {
			console.log(response.history.loss[0]);
			lossovertime.push(response.history.loss[0]);
		})
		}
}


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

const sgdOptimizer = tf.train.rmsprop(learningRate);

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



	

	




train().then(() => { 
	console.log("Training Complete.");
	let outputs = model.predict(xs);
	outputs.print();
	
	//chart:
let labels = [];
for (let i = 0; i<lossovertime.length; i++){
	labels.push(i+1);
}
let ctx = document.getElementById("chartContainer").getContext('2d');
let myChart = new Chart(ctx, {
    type: 'line',
    data: {
		labels: labels,
        datasets: [{
            label: 'Loss Overtime',
            data: lossovertime,
            backgroundColor: [
                'rgba(255, 99, 132, 0.3)'
            ],
            borderColor: [
                'rgba(255,99,132,1)'
            ],
            borderWidth: 1
        }]
    },
    options: {
        scales: {
            yAxes: [{
                ticks: {
                    beginAtZero:false
                }
            }]
        }
    }
});
outputs.data().then(() => {
	let output = document.getElementById("output");
	let expected = document.getElementById("expected");
	expected.innerHTML = "";
	output.innerHTML = "";
	
	expected.innerHTML += "Expected: <br>"
	expected.innerHTML += "0.2 , 0.2 , 0.2<br>0.2 , 0.2 , 0.2<br>0.2 , 0.2 , 0.2"
	output.innerHTML += "Output Tensor: <br>";
	
	
	for (let i = 1 ; i <= 9 ; i++ ) {
		output.innerHTML += Math.round(outputs.dataSync()[i-1]*10000)/10000;
		if (i%3 == 0 && i != 0) {
			output.innerHTML += " <br> ";
		}
		else if (i != 9 ){
			output.innerHTML += " , ";
		}
		
	}
 	
});

	
	
})




}









