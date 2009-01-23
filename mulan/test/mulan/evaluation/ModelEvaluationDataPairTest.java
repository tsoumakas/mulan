package mulan.evaluation;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Test routines for {@link ModelEvaluationDataPair} class.
 * 
 * @author Jozef Vilcek
 */
public class ModelEvaluationDataPairTest {

	private static final List<Boolean> MODEL_OUTPUT = 
		new ArrayList<Boolean>(Arrays.asList(new Boolean[] { true, true, false, false }));
	private static final List<Boolean> ACTUAL_OUTPUT = 
		new ArrayList<Boolean>(Arrays.asList(new Boolean[] { false, true, true, false }));
	private ModelEvaluationDataPair<Boolean> dataPair;
	
	@Before
	public void setUp(){
		dataPair = new ModelEvaluationDataPair<Boolean>(MODEL_OUTPUT, ACTUAL_OUTPUT);
	}
	
	@After
	public void tearDown(){
		dataPair = null;
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithNullParams(){
		new ModelEvaluationDataPair<Boolean>(null, null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithNullModelOut(){
		new ModelEvaluationDataPair<Boolean>(null, ACTUAL_OUTPUT);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithNullActualOut(){
		new ModelEvaluationDataPair<Boolean>(MODEL_OUTPUT, null);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithWrongConfidences(){
		dataPair = new ModelEvaluationDataPair<Boolean>(MODEL_OUTPUT, ACTUAL_OUTPUT, new ArrayList<Double>());
	}
	
	@Test
	public void testGetModelOutput(){
		Boolean[] result = dataPair.getModelOutput().toArray(new Boolean[]{});
		assertTrue("Returned model output has incorrect values.", 
				Arrays.equals(MODEL_OUTPUT.toArray(new Boolean[]{}), result));
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testGetModelOutputIsReadOnly(){
		dataPair.getModelOutput().add(true);
	}
	
	@Test
	public void testGetActualOutput(){
		Boolean[] result = dataPair.getActualOutput().toArray(new Boolean[]{});
		assertTrue("Returned actual output has incorrect values.", 
				Arrays.equals(ACTUAL_OUTPUT.toArray(new Boolean[]{}), result));
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testGetActualOutputIsReadOnly(){
		dataPair.getActualOutput().add(true);
	}
	
	@Test
	public void testGetConfidencesAreNull(){
		assertNull("Model confidences should return null.", dataPair.getModelConfidences()); 
	}
	
	@Test
	public void testGetConfidences(){
		List<Double> conficences = new ArrayList<Double>();
		conficences.addAll(Arrays.asList(new Double[]{0.1, 0.9, -0.9, -0.3}));
		ModelEvaluationDataPair<Boolean> dataPair = 
			new ModelEvaluationDataPair<Boolean>(MODEL_OUTPUT, ACTUAL_OUTPUT, conficences);
		
		assertNotNull("Confidences should not be null.", dataPair.getModelConfidences());
		Double[] result = dataPair.getModelConfidences().toArray(new Double[]{});
		assertTrue("Returned model confidences has incorrect values.", 
				Arrays.equals(conficences.toArray(new Double[]{}), result));
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testGetModelConfidencesIsReadOnly(){
		List<Double> conficences = new ArrayList<Double>();
		conficences.addAll(Arrays.asList(new Double[]{0.1, 0.9, -0.9, -0.3}));
		ModelEvaluationDataPair<Boolean> dataPair = 
			new ModelEvaluationDataPair<Boolean>(MODEL_OUTPUT, ACTUAL_OUTPUT, conficences);
		
		dataPair.getModelConfidences().add(Double.MAX_VALUE);
	}
	
		
}
