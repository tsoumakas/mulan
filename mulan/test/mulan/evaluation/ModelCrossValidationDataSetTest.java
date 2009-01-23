package mulan.evaluation;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Test routines for {@link ModelCrossValidationDataSet} class.
 * 
 * @author Jozef Vilcek
 */
public class ModelCrossValidationDataSetTest {

	private static final int FOLD_NUMBER = 1;
	private static final List<ModelEvaluationDataPair<Boolean>> DATA_ONE = 
		new ArrayList<ModelEvaluationDataPair<Boolean>>();
	private static final List<ModelEvaluationDataPair<Boolean>> DATA_TWO = 
		new ArrayList<ModelEvaluationDataPair<Boolean>>();

	private ModelCrossValidationDataSet<Boolean> dataSet;
	
	@Before
	public void setUp(){
		dataSet = new ModelCrossValidationDataSet<Boolean>();
	}
	
	@After
	public void tearDown(){
		dataSet = null;
	}
	
	@Test
	public void testConstructor(){
		assertEquals("Data set should be empty.", 0, dataSet.getNumFolds());
	}

	@Test
	public void testAddGetFoldData(){
		dataSet.addFoldData(FOLD_NUMBER, DATA_ONE);
		assertSame("Data returned for specified fold are invalid.", 
				DATA_ONE, dataSet.getFoldData(FOLD_NUMBER));
		
		dataSet.addFoldData(FOLD_NUMBER, DATA_TWO);
		assertSame("Data returned for specified fold are invalid.", 
				DATA_TWO, dataSet.getFoldData(FOLD_NUMBER));
	}
	
	@Test
	public void testRemoveFoldData(){
		dataSet.addFoldData(FOLD_NUMBER, DATA_ONE);
		assertNotNull("Data returned for specified fold should not be null.", 
				dataSet.getFoldData(FOLD_NUMBER));
		
		dataSet.removeFoldData(FOLD_NUMBER);
		assertNull("Data returned for specified fold should be null.", 
				dataSet.getFoldData(FOLD_NUMBER));
	}
	
	
	
		
}
