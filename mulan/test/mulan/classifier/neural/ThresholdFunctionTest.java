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
package mulan.classifier.neural;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class ThresholdFunctionTest {

	private static final double DOUBLES_EQUAL_DIFF = 0.000001;
	private static final int NUM_LABELS = 3;
	private static final double[][] IDEAL_LABELS =	new double[][] {{-1, 1, 1}, 
																	{1, -1, -1}, 
																	{1, -1, 1}, 
																	{-1, -1, 1}};
	private static final double[][] LABEL_CONFIDENCES = new double[][] {{-0.8, 0.5, 0.6}, 
																		{0.9, -0.8, -0.4}, 
																		{0.7, -0.6, 0.9}, 
																		{-0.5, -0.6, 0.8}};
	private static final double THRESHOLD_FOR_EXAMPLE_1 = -0.15;
	private static final double THRESHOLD_FOR_EXAMPLE_2 = 0.25;
	private static final double THRESHOLD_FOR_EXAMPLE_3 = 0.05;
	private static final double THRESHOLD_FOR_EXAMPLE_4 = 0.15;
	
	private ThresholdFunction function;
	
	
	@Before
	public void setUp(){
		function = new ThresholdFunction(IDEAL_LABELS, LABEL_CONFIDENCES);
	}
	
	@After
	public void tearDown(){
		function = null;
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithNullIdealLabels(){
		new ThresholdFunction(null, LABEL_CONFIDENCES);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithNullConfidences(){
		new ThresholdFunction(IDEAL_LABELS, null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithWrongDimensionsOfInputs1(){
		new ThresholdFunction(new double[3][4], new double[2][4]);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithWrongDimensionsOfInputs2(){
		new ThresholdFunction(new double[3][4], new double[3][3]);
	}
	
	@Test
	public void testConstructor(){
		assertNotNull("Threshold function should be built after creating new instance.", function.getFunctionParameters());
	}
	
	@Test
	public void testBuild(){
		function.build(IDEAL_LABELS, LABEL_CONFIDENCES);
		
		double[] parameters = function.getFunctionParameters();
		assertNotNull("Parameters learned by the function should not be null.", function.getFunctionParameters());
		assertTrue("Parameters learned by the function are not as expected.", parameters.length == NUM_LABELS + 1);
	}
	
	@Test
	public void testComputeThreshold(){
		double threshold = function.computeThreshold(LABEL_CONFIDENCES[0]);
		assertEquals("Computed threshold is not as expected.", THRESHOLD_FOR_EXAMPLE_1, threshold, DOUBLES_EQUAL_DIFF);
		threshold = function.computeThreshold(LABEL_CONFIDENCES[1]);
		assertEquals("Computed threshold is not as expected.", THRESHOLD_FOR_EXAMPLE_2, threshold, DOUBLES_EQUAL_DIFF);
		threshold = function.computeThreshold(LABEL_CONFIDENCES[2]);
		assertEquals("Computed threshold is not as expected.", THRESHOLD_FOR_EXAMPLE_3, threshold, DOUBLES_EQUAL_DIFF);
		threshold = function.computeThreshold(LABEL_CONFIDENCES[3]);
		assertEquals("Computed threshold is not as expected.", THRESHOLD_FOR_EXAMPLE_4, threshold, DOUBLES_EQUAL_DIFF);
	}
}
