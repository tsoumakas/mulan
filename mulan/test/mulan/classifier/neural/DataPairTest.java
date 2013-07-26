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

import java.util.Arrays;

import mulan.core.ArgumentNullException;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Test routines for {@link DataPair} class.
 * 
 * @author Jozef Vilcek
 */
public class DataPairTest {

	private static final double[] TEST_INPUT = new double[]{0.1, 0.2, 3.3, 3.4};
	private static final double[] TEST_OUTPUT = new double[]{1, 2, 3};
	private DataPair dataPair;
	
	@Before
	public void setUp(){
		dataPair = new DataPair(TEST_INPUT, TEST_OUTPUT);
	}
	
	@After
	public void tearDown(){
		dataPair = null;
	}
	
//	@Test(expected=ArgumentNullException.class)
//	public void testConstructorWithNullParams(){
//		new DataPair(null, null);
//	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructorWithNullParam1(){
		new DataPair(null, TEST_OUTPUT);
	}
	
	@Test(expected=ArgumentNullException.class)
	public void testConstructorWithNullParam2(){
		new DataPair(TEST_INPUT, null);
	}
	
	@Test
	public void testImutable(){
		assertNotSame("The input returned by DataPair should not be same instance as input param used " +
				"for constructor.", TEST_INPUT, dataPair.getInput());
		assertNotSame("The output returned by DataPair should not be same instance as output param used " +
				"for constructor.", TEST_OUTPUT, dataPair.getInput());
	}
	
	@Test
	public void testGetInput(){
		double[] result = dataPair.getInput();
		assertTrue("Returned array is not as expectated.", Arrays.equals(TEST_INPUT, result));
	}
	
	@Test
	public void testGetOutput(){
		double[] result = dataPair.getOutput();
		assertTrue("Returned array is not as expectated.", Arrays.equals(TEST_OUTPUT, result));
	}
}
