/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */
package mulan.classifier.neural.model;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import mulan.core.ArgumentNullException;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

public class BasicNeuralNetTest {

	private static final double BIAS = 1;
	private static final int[] NET_TOPOLOGY = new int[] { 2, 10, 3 };
	private static final double[] INPUT_PATTERN = new double[] { 1.0, -1.0 };
	private Class<ActivationTANH> ACT_FUNCT_CLASS = ActivationTANH.class;
	private Class<ActivationLinear> ACT_FUNCT_INPUT_LAYER_CLASS = ActivationLinear.class;
	private BasicNeuralNet neuralNet;
	
	@Before
	public void setUp(){
		neuralNet = new BasicNeuralNet(NET_TOPOLOGY, BIAS, ACT_FUNCT_CLASS, null);
	}
	
	@After
	public void tearDown(){
		neuralNet = null;
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithNullTolopogy(){
		new BasicNeuralNet(null, BIAS, ACT_FUNCT_INPUT_LAYER_CLASS, null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithWrongTolology(){
		new BasicNeuralNet(new int[] {0}, BIAS, ACT_FUNCT_INPUT_LAYER_CLASS, null);
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructorWithNullActFunction(){
		new BasicNeuralNet(NET_TOPOLOGY, BIAS, null, null);
	}
	
	@Test
	public void testGetLayersCount(){
		assertEquals("Count of neural network layers is not as expected.", NET_TOPOLOGY.length , neuralNet.getLayersCount());
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testGetLayerUnitsUnmodifiableAdd(){
		List<Neuron> layer = neuralNet.getLayerUnits(0);
		layer.add(null);
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testGetLayerUnitsUnmodifiableRemove(){
		List<Neuron> layer = neuralNet.getLayerUnits(0);
		layer.remove(0);
	}
	
	@Test
	public void testGetLayerUnits(){
		for(int layerIndex = 0; layerIndex < NET_TOPOLOGY.length; layerIndex++){
			List<Neuron> layer = neuralNet.getLayerUnits(layerIndex);
			assertEquals("Count of units of a layer is not as expected.", 
					NET_TOPOLOGY[layerIndex], layer.size());
			
			for(Neuron neuron : layer){
				assertEquals("Neuron bias is not as expected.", BIAS, neuron.getBiasInput(), 0);
				if(layerIndex == 0){
					assertEquals("Activation function of a neuron is not as expected.", 
							ACT_FUNCT_INPUT_LAYER_CLASS, neuron.getActivationFunction().getClass());
				}
				else{
					assertEquals("Activation function of a neuron is not as expected.", 
							ACT_FUNCT_CLASS, neuron.getActivationFunction().getClass());
				}
			}
		}
	}
	
	@Test
	public void testGetOutput(){
		double[] zeroArray = new double[NET_TOPOLOGY[NET_TOPOLOGY.length - 1]];
		Arrays.fill(zeroArray, 0);
		
		assertTrue("Network output is not as expected.", Arrays.equals(zeroArray, neuralNet.getOutput()));
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testFeedForwardWithNull(){
		neuralNet.feedForward(null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testFeedForwardWithWrongInput(){
		neuralNet.feedForward(new double[0]);
	}
	
	@Test
	public void testFeedForward(){
		double[] result = neuralNet.feedForward(INPUT_PATTERN);
		
		assertTrue("Network output is not as expected.", Arrays.equals(result, neuralNet.getOutput()));
		double sum = 0;
		for(double item : result){
			sum += item;
		}
		assertFalse("The output of the network after processing input shoud not be zero.", sum == 0);
		assertTrue("The output of the network should be same as in previous feedForward call.", Arrays.equals(result, neuralNet.feedForward(INPUT_PATTERN)));
	}
	
	@Test
	public void testReset(){
		testFeedForward();
		neuralNet.reset();
		testGetOutput();
	}
	
	@Test
	public void testGetNetInputSize(){
		assertEquals("The size of the network input is not as expected.", 
				NET_TOPOLOGY[0], neuralNet.getNetInputSize());
	}
	
	@Test
	public void testGetNetOutputSize(){
		assertEquals("The size of the network output is not as expected.", 
				NET_TOPOLOGY[NET_TOPOLOGY.length - 1], neuralNet.getNetOutputSize());
	}
	
	@Test
	public void testRandomness(){
		neuralNet = new BasicNeuralNet(NET_TOPOLOGY, BIAS, ACT_FUNCT_CLASS, new Random(10));
		double[] weights = neuralNet.getLayerUnits(1).get(0).getWeights();
		neuralNet = new BasicNeuralNet(NET_TOPOLOGY, BIAS, ACT_FUNCT_CLASS, new Random(10));
		Assert.assertTrue(Arrays.equals(weights, neuralNet.getLayerUnits(1).get(0).getWeights()));
		
		neuralNet = new BasicNeuralNet(NET_TOPOLOGY, BIAS, ACT_FUNCT_CLASS, null);
		weights = neuralNet.getLayerUnits(1).get(0).getWeights();
		neuralNet = new BasicNeuralNet(NET_TOPOLOGY, BIAS, ACT_FUNCT_CLASS, null);
		Assert.assertFalse(Arrays.equals(weights, neuralNet.getLayerUnits(1).get(0).getWeights()));
	}

}
