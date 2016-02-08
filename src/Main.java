import weka.core.Instances;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.J48;
import weka.filters.unsupervised.attribute.Remove;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;

public class Main {

	public static void main(String[] args) {
		try {
			Instances dataTrain = load("spamdata-dev-valid.arff");
			Instances dataTest = load(args[0]);

			export(predict(j48(dataTrain, false), dataTest), args[1]);
			export(predict(ibk(dataTrain), dataTest), args[2]);

			
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}
	
	public static void export(String content, String fileName) throws FileNotFoundException {
		PrintWriter out = new PrintWriter(fileName);
	    out.print(content);
	    out.close();
	}
	
	public static Instances load(String arff) throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader("src/arff/"+arff));
		Instances data = new Instances(reader);
		reader.close();
		data.setClassIndex(data.numAttributes() - 1);
		
		return data;
	}
	
	public static String predict(FilteredClassifier fc, Instances test) throws Exception {
		String content = "";
		for (int i = 0; i < test.numInstances(); i++) {
			double pred = fc.classifyInstance(test.instance(i));
			content += test.classAttribute().value((int) pred)+"\n";
		}
		return content;
	}
	
	public static FilteredClassifier j48(Instances train, boolean unpruned) throws Exception {
		J48 j48 = new J48();
		j48.setUnpruned(unpruned);
		FilteredClassifier fc = new FilteredClassifier();
		
		Remove rm = new Remove();
		rm.setAttributeIndices("1");  // remove 1st attribute
		
		fc.setFilter(rm);
		fc.setClassifier(j48);
		fc.buildClassifier(train);
		return fc;
	}
	
	public static FilteredClassifier naiveBaye(Instances train) throws Exception {
		NaiveBayes naiveBaye = new NaiveBayes();
		FilteredClassifier fc = new FilteredClassifier();
		
		Remove rm = new Remove();
		rm.setAttributeIndices("1");  // remove 1st attribute
		
		fc.setFilter(rm);
		fc.setClassifier(naiveBaye);
		fc.buildClassifier(train);
		return fc;
	}
	
	public static FilteredClassifier ibk(Instances train) throws Exception {
		IBk ibk = new IBk();
		FilteredClassifier fc = new FilteredClassifier();
		
		Remove rm = new Remove();
		rm.setAttributeIndices("1");  // remove 1st attribute
		
		fc.setFilter(rm);
		fc.setClassifier(ibk);
		fc.buildClassifier(train);
		return fc;
	}
}
