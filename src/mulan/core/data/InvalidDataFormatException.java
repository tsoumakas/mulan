package mulan.core.data;

import mulan.core.MulanException;

/**
 * The exception is thrown when format of the data is not valid.
 * 
 * @author Jozef Vilcek
 */
public class InvalidDataFormatException extends MulanException {

	private static final long serialVersionUID = -8323657086903118700L;

	/**
	 * Creates a new instance of {@link InvalidDataFormatException} with detail mesage.
	 * @param message the detail message
	 */
	public InvalidDataFormatException(String message){
		super(message);
	}
	
	/**
	 * Creates a new instance of {@link InvalidDataFormatException} with detail message 
	 * and nested exception.
	 * 
	 * @param message the detail message
	 * @param cause the nested exception
	 */
	public InvalidDataFormatException(String message, Throwable cause){
		super(message, cause);
	}
}
