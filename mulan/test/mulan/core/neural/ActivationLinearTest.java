package mulan.core.neural;

import static org.junit.Assert.*;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class ActivationLinearTest {

	private static final double EXPECTED_FUNCT_MAX = Double.POSITIVE_INFINITY;
	private static final double EXPECTED_FUNCT_MIN = Double.NEGATIVE_INFINITY;
	private static final double[] INPUT_VALUES = { Double.NEGATIVE_INFINITY, -10.99, 0, 
												 10.99, Double.POSITIVE_INFINITY };
	private static final double[] EXPECTED_FUNCT_OUT_VALUES = { Double.NEGATIVE_INFINITY, -10.99, 0, 
		 												 10.99, Double.POSITIVE_INFINITY };
	
	private ActivationLinear function;
	
	@Before
	public void setUp(){
		function = new ActivationLinear();
	}
	
	@After
	public void tearDown(){
		function = null;
	}
	
	@Test
	public void testActivate(){
		for(int i = 0; i < INPUT_VALUES.length; i++){
			double functOutput = function.activate(INPUT_VALUES[i]);
			assertEquals("The function output is not as expected.", EXPECTED_FUNCT_OUT_VALUES[i], functOutput, 0);			
		}
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testDerivative(){
		function.derivative(0);
	}
	
	@Test
	public void testGetMax(){
		double max = function.getMax();
		assertEquals("Maximu of the function is not as expected.", EXPECTED_FUNCT_MAX, max, 0);
	}
	
	@Test
	public void testGetMin(){
		double min = function.getMin();
		assertEquals("Minimum of the function is not as expected.", EXPECTED_FUNCT_MIN, min, 0);
	}
}
