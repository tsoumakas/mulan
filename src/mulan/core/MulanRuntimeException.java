package mulan.core;

/**
 * Represents a root base class for unchecked exceptions thrown within Mulan library.
 * 
 * @author Jozef Vilcek
 */
public class MulanRuntimeException extends RuntimeException {

	private static final long serialVersionUID = -1043393739740022098L;

	/**
	 * Creates a new instance of {@link MulanRuntimeException} with the specified
	 * detail message. 
	 * 
	 * @param message the detail message
	 */
	public MulanRuntimeException(String message){
		super(message);
	}
	
	/**
	 * Creates a new instance of {@link MulanRuntimeException} with the specified
	 * detail message and nested exception.
	 * 
	 * @param message the detail message
	 * @param cause the nested exception
	 */
	public MulanRuntimeException(String message, Throwable cause){
		super(message, cause);
	}

}
