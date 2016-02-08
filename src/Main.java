import weka.core.Instances;
import weka.classifiers.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Main {

	public static void main(String[] args) {
		try {
			BufferedReader reader = new BufferedReader(new FileReader("src/arff/spamdata-dev-train.arff"));
			Instances data = new Instances(reader);
			reader.close();
			data.setClassIndex(data.numAttributes() - 1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		// setting class attribute
		
	}
}
