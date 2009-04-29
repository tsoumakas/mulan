package mulan.core;

/**
 * Represents a root base class for checked exceptions thrown within Mulan library.
 * 
 * @author Jozef Vilcek
 */
public class MulanException extends Exception {

	private static final long serialVersionUID = 2271544759439172440L;

	/**
	 * Creates a new instance of {@link MulanException} with the specified
	 * detail message. 
	 * 
	 * @param message the detail message
	 */
	public MulanException(String message){
		super(message);
	}
	
	/**
	 * Creates a new instance of {@link MulanException} with the specified
	 * detail message and nested exception.
	 * 
	 * @param message the detail message
	 * @param cause the nested exception
	 */
	public MulanException(String message, Throwable cause){
		super(message, cause);
	}

}
