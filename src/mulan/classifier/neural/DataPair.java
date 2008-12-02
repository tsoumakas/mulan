
package mulan.classifier.neural;

import java.util.Arrays;
	
/**
 * Class for holding data pair for neural network. 
 * The data pair contains the input pattern and respected 
 * expected/ideal network output/response pattern for the input.
 * 
 * @author Jozef Vilcek
 */
public class DataPair {

	private final double[] input;
	private final double[] output;
	
	/**
	 * Creates a {@link DataPair} instance.
	 * @param input the input pattern
	 * @param output the ideal/expected output pattern for the input
	 */
	public DataPair(final double[] input, final double[] output){
		if(input == null || output == null){
			throw new IllegalArgumentException("Failed to create an instance. Either input or output pattern is null.");
		}
		this.input = Arrays.copyOf(input, input.length);
		this.output = Arrays.copyOf(output, output.length);
	}
	
	/**
	 * Gets the input pattern.
	 * @return the input pattern
	 */
	public double[] getInput(){
		return input;
	}
	
	/**
	 * Gets the idel/expected output pattern.
	 * @return the output pattern
	 */
	public double[] getOutput(){
		return output;
	}
}
	

