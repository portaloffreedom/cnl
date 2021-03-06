#pragma once

class InputTestSet;

class InputTest
{
	//friend class MLP;
	friend class InputTestSet;
	friend class CUDATools;

	vector<double> m_vecInputs;
	//real correctOutput;
	//real networkOutput;
	vector<double> m_vecCorrectOutputs;
	vector<double> m_vecNetworkOutputs;
	vector<double> m_vecNetworkOutputsGPU; // But values on a GPU may be float!

	InputTestSet *m_pParentTestSet;

	InputTest(InputTestSet *p_pParentTestSet, unsigned p_uNumberInputs,unsigned p_uNumberOutputs);

	void randomizeTest(MTRand *p_pRandomGenerator);
	void setOutputFunction(const vector< pair<double,double> > &p_vecMinMax, void (*p_fTestingFunction)(const vector<double> &p_vecInputParameters,vector<double> &p_vecOutputParameters),MTRand *p_pRandomGenerator);

	void saveDoubleTestVectorToXML(const vector<double> &p_vecDoubleValues,TiXmlElement &p_XML,Str p_sNameToSave,bool p_bOutputAttribute) const;
	void saveToXML(TiXmlElement &p_XML) const;
	static void loadDoubleTestVectorFromXML(vector<double> &p_vecDoubleValues,const TiXmlElement &p_XML,Str p_sNameToLoad);
	void loadFromXML(const TiXmlElement &p_XML);

public:
	void setOutputs(const vector<double> &p_vecLayerOutput);
	const vector<double>& getInputs() const;
	double getNetworkOutput(unsigned int p_uiIndex) const;
	double getCorrectOutput(unsigned int p_uiIndex) const;
};
