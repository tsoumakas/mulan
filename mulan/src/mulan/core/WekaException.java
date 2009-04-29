package mulan.core;


/**
 * The convenience exception, which can be used to wrap up checked general {@link Exception}
 * commonly thrown by underlying Weka library into anonymous runtime exception.
 * <br></br><br></br> 
 * Note: The preferred way of handling Weka exceptional states is to define custom typed 
 * exception thrown by Mulan, which specifies a context about failure reason. 
 * 
 * @author Jozef Vilcek
 */
public class WekaException extends MulanRuntimeException {

	private static final long serialVersionUID = -8041689691825060987L;

	/**
	 * Creates a new instance of {@link WekaException} with detail mesage.
	 * @param message the detail message
	 */
	public WekaException(String message){
		super(message);
	}
	
	/**
	 * Creates a new instance of {@link WekaException} with detail message 
	 * and nested exception.
	 * 
	 * @param message the detail message
	 * @param cause the nested exception
	 */
	public WekaException(String message, Throwable cause){
		super(message, cause);
	}
}
