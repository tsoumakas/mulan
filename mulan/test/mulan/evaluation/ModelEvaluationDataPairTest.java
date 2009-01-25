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

	private static final Boolean MODEL_OUTPUT = true;
	private static final List<Boolean> TRUE_LABELS = 
		new ArrayList<Boolean>(Arrays.asList(new Boolean[] { false, true, true, false }));
	private ModelEvaluationDataPair<Boolean> dataPair;
	
	@Before
	public void setUp(){
		dataPair = new ModelEvaluationDataPair<Boolean>(MODEL_OUTPUT, TRUE_LABELS);
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
		new ModelEvaluationDataPair<Boolean>(null, TRUE_LABELS);
	}
	
	@Test(expected=IllegalArgumentException.class)
	public void testConstructorWithNullActualOut(){
		new ModelEvaluationDataPair<Boolean>(MODEL_OUTPUT, null);
	}
			
	@Test
	public void testGetModelOutput(){
		Boolean result = dataPair.getModelOutput();
		assertSame("Returned model output has incorrect values.", MODEL_OUTPUT, result);
	}
	
	public void testGetModelOutputIsReadOnly(){
		Boolean result = dataPair.getModelOutput();
		assertSame("Model output is not as expected.",MODEL_OUTPUT, result);
		result = new Boolean(false);
		assertSame("Model output is not as expected.",MODEL_OUTPUT, dataPair.getModelOutput());
	}
	
	@Test
	public void testGetTrueLabels(){
		Boolean[] result = dataPair.getTrueLabels().toArray(new Boolean[]{});
		assertTrue("Returned actual output has incorrect values.", 
				Arrays.equals(TRUE_LABELS.toArray(new Boolean[]{}), result));
	}
	
	@Test(expected=UnsupportedOperationException.class)
	public void testGetActualOutputIsReadOnly(){
		dataPair.getTrueLabels().add(true);
	}
}
