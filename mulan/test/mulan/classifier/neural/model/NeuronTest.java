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

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import mulan.core.ArgumentNullException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class NeuronTest {

	private static final double DOUBLES_EQUAL_DELTA = 0.000001;
	private static final double WEIGHTS_MAX_VALUE = 1.0;
	private static final double WEIGHTS_MIN_VALUE = -1.0;
	private static final double TEST_DOUBLE_VALUE = 9.9;
	private static final int INPUT_DIM = 5;
	private static final double WEIGHTS_VALUE_FOR_PROCESS = 1.0;
	private static final double[] INPUT_PATTERN_TO_PROCESS = new double[]{ 0.1, 0.2, -0.3, 0.2, 0.1 };
	private static final double EXPECTED_OUT_FROM_PROCESS = 0.861723; 
	private static final double NEURON_BIAS = 1;
	private static final int INVALID_INPUT_DIM = 0;
	private static final ActivationFunction ACTIVATION_FUNCTION = new ActivationTANH();
	
	private Neuron neuron;
	
	@Before
	public void setUp(){
		neuron = new Neuron(ACTIVATION_FUNCTION, INPUT_DIM, NEURON_BIAS);
	}
	
	@After
	public void tearDown(){
		neuron = null;
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructorWithNullActivationFunction(){
		new Neuron(null, INPUT_DIM, NEURON_BIAS);
	}
		
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithInvalidDim(){
		new Neuron(ACTIVATION_FUNCTION, INVALID_INPUT_DIM, NEURON_BIAS);
	}
	
	@Test
	public void testConstructor(){
		verifyNeuronResetState(neuron);
	}
	
	@Test
	public void testGetActivationFunction(){
		assertEquals("Activation function is not as expected.", 
				ACTIVATION_FUNCTION, neuron.getActivationFunction());
	}
	
	@Test
	public void testGetWeigts(){
		double[] weights = neuron.getWeights();
		assertTrue(weights[0] != TEST_DOUBLE_VALUE);
		weights[0] = TEST_DOUBLE_VALUE;
		assertEquals("Neuron weight does not have expected value", 
				TEST_DOUBLE_VALUE, neuron.getWeights()[0], 0);
	}
	
	@Test
	public void testGetDeltas(){
		double[] deltas = neuron.getDeltas();
		assertTrue(deltas[0] != TEST_DOUBLE_VALUE);
		deltas[0] = TEST_DOUBLE_VALUE;
		assertEquals("Neuron delta does not have expected value", 
				TEST_DOUBLE_VALUE, neuron.getDeltas()[0], 0);
	}
	
	@Test
	public void testSetError(){
		double error = neuron.getError();
		assertTrue(error != TEST_DOUBLE_VALUE);
		neuron.setError(TEST_DOUBLE_VALUE);
		assertEquals("Neuron error does not have expected value", 
				TEST_DOUBLE_VALUE, neuron.getError(), 0);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testProcessInputWithNull(){
		neuron.processInput(null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testProcessInputWithWrongPatternDimension(){
		neuron.processInput(new double[] { 0 });
	}
	
	@Test
	public void testProcessInput(){
		Arrays.fill(neuron.getWeights(), WEIGHTS_VALUE_FOR_PROCESS);
		
		double result = neuron.processInput(INPUT_PATTERN_TO_PROCESS);
		assertEquals("Output of neuron is not as expected.", 
				EXPECTED_OUT_FROM_PROCESS, result, DOUBLES_EQUAL_DELTA);
	}
	
	@Test
	public void testReset(){
		neuron.setError(TEST_DOUBLE_VALUE);
		boolean failFlag = false;
		try{
			verifyNeuronResetState(neuron);
			failFlag = true;
		}
		catch(AssertionError error){}
		if(failFlag){
			fail("Neuron should not be in state equal to reset.");
		}
		
		neuron.reset();
		verifyNeuronResetState(neuron);
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testAddNeuronWithNull(){
		neuron.addNeuron(null);
	}
	
	@Test
	public void testAddNeuron(){
		final int initialConnectedNeuronsCount = 0;
		final int finalConnectedNeuronsCount = 1;
		assertEquals("Initial count of connected neurons is not as expected.", 
				initialConnectedNeuronsCount, neuron.getConnectedNeuronsCount());
		Neuron connectedNeuron = new Neuron(ACTIVATION_FUNCTION, INPUT_DIM, NEURON_BIAS);
		boolean result = neuron.addNeuron(connectedNeuron);
		assertTrue("The neuron should be added.", result);
		assertEquals("The count of connected neurons is not as expected.", 
				finalConnectedNeuronsCount, neuron.getConnectedNeuronsCount());
		
		result = neuron.addNeuron(connectedNeuron);
		assertFalse("The neuron should not be added.", result);
		assertEquals("The count of connected neurons is not as expected.", 
				finalConnectedNeuronsCount, neuron.getConnectedNeuronsCount());
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testRemoveNeuronWithNull(){
		neuron.removeNeuron(null);
	}
	
	@Test
	public void testRemoveNeuron(){
		final int expectedCountAfterRemove = 0;
		
		Neuron connectedNeuron = new Neuron(ACTIVATION_FUNCTION, INPUT_DIM, NEURON_BIAS);
		neuron.addNeuron(connectedNeuron);
		
		boolean result = neuron.removeNeuron(connectedNeuron);
		assertTrue("The neuron should be removed.", result);
		assertEquals("The count of connected neurons is not as expected.", 
				expectedCountAfterRemove, neuron.getConnectedNeuronsCount());
		
		result = neuron.removeNeuron(connectedNeuron);
		assertFalse("The neuron should not be removed.", result);
		assertEquals("The count of connected neurons is not as expected.", 
				expectedCountAfterRemove, neuron.getConnectedNeuronsCount());
	}
	
	@Test(expected = IllegalArgumentException.class)
	public void testAddAllNeuronsWithNull(){
		neuron.addAllNeurons(null);
	}
	
	@Test
	public void testAddAllNeurons(){
		List<Neuron> neurons = new ArrayList<Neuron>();
		neurons.add(new Neuron(ACTIVATION_FUNCTION, INPUT_DIM, NEURON_BIAS));
		neurons.add(new Neuron(ACTIVATION_FUNCTION, INPUT_DIM, NEURON_BIAS));
		
		boolean result = neuron.addAllNeurons(neurons);
		assertTrue("At least one neuron should be added.", result);
		assertEquals("The count of connected neurons is not as expected.", 2, neuron.getConnectedNeuronsCount());
		
		result = neuron.addAllNeurons(neurons);
		assertFalse("None of neurons should be added.", result);
		assertEquals("The count of connected neurons is not as expected.", 2, neuron.getConnectedNeuronsCount());
		
		neurons.add(new Neuron(ACTIVATION_FUNCTION, INPUT_DIM, NEURON_BIAS));
		result = neuron.addAllNeurons(neurons);
		assertTrue("At lest one neuron should be added.", result);
		assertEquals("The count of connected neurons is not as expected.", 3, neuron.getConnectedNeuronsCount());
	}
	
	private void verifyNeuronResetState(Neuron neuron){
		assertEquals("Neuron error does not have expected value.", 0, neuron.getError(), 0);
		assertEquals("Neuron input does not have expected value.", 0, neuron.getNeuronInput(), 0);
		assertEquals("Neuron output does not have expected value.", 0, neuron.getOutput(), 0);
		double[] deltas = neuron.getDeltas();
		for(double item : deltas){
			assertEquals("Neuron delta does not have expected value.", 0, item, 0);
		}
		double[] weights = neuron.getWeights();
		for(double item : weights){
			assertTrue("Neuron weight not properly initialized.", 
					item < WEIGHTS_MAX_VALUE && item > WEIGHTS_MIN_VALUE);
		}
	}
	
}
