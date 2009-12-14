#include "..\Global\stdafx.h"

__constant__ int iTestIndices[iMaxNumberOfTrainedElements];

//JRTODO - dylemat - czy zawsze uzywac iTestIndices przy czytaniu i zapisywaniu, czy tylko na wejsciu pierwszego layera?..
// JRTDO - zmien mnozenie integerow na specjalna funkcje mnozaca najnizsze 24 bity

__global__ void executeLayerKernel(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput,int p_iNumInputNeurons
								   ,int p_iNumInputNeuronsAligned, Neuron::NeuronType p_eNeuronType,int p_iOutputNeuronCount,bool p_bInTraining,int p_iHowMuchMemoryForWeights)
{
	extern __shared__ real_gpu s_InputNeurons[];
	real_gpu* s_InputWeights = &s_InputNeurons[p_iNumInputNeurons];

	int iTestIndex;
	if(p_bInTraining)
		iTestIndex = iTestIndices[blockIdx.x];
	else
		iTestIndex = blockIdx.x;
	
	const real_gpu *d_LayerInputThisTest = dp_pLayerInput + iTestIndex*p_iNumInputNeuronsAligned;
	int iMoveWeightsForThisTest = threadIdx.x*p_iNumInputNeurons;
	const real_gpu *d_WeightsThisTest = dp_pWeights + iMoveWeightsForThisTest;
	real_gpu *d_pLayerOutputThisTest = dp_pLayerOutput + blockIdx.x*blockDim.x + threadIdx.x;
	real_gpu *d_pDerivativeOfLastOutputThisTest = dp_pDerivativeOfLastOutput + blockIdx.x*blockDim.x + threadIdx.x;

	// first, we copy d_LayerInputThisTest to s_InputNeurons
	for(int iInputIndex = threadIdx.x;iInputIndex < p_iNumInputNeurons; iInputIndex+=blockDim.x)
	{
		s_InputNeurons[iInputIndex] = d_LayerInputThisTest[iInputIndex];
	}

	// we have to make sure that all data was written to shared memory
	__syncthreads();

	real_gpu dResult = 0.0f;
	
	//if(threadIdx.x == 1 && blockIdx.x == 1)
	//{
	//	PRINT_DEBUG_INFO("BX %d TX %d | INPUT %d | WEIGHTS %d | OUTPUT %d\n",blockIdx.x,threadIdx.x,d_LayerInputThisTest - dp_pLayerInput,d_WeightsThisTest - dp_pWeights,d_pLayerOutputThisTest - dp_pLayerOutput);
	//}

	int iNumOfWeights = p_iNumInputNeurons * p_iOutputNeuronCount;
	int iNumOfWeightsAligned = ALIGN_UP(iNumOfWeights,blockDim.x);
	for(int iWeightIndex = threadIdx.x, iWeightIndexBase = 0 ; iWeightIndex < iNumOfWeightsAligned ; iWeightIndex += p_iHowMuchMemoryForWeights, iWeightIndexBase += p_iHowMuchMemoryForWeights)
	{
		/*if(threadIdx.x == 0)
		{
			PRINT_DEBUG_INFO("GPU: NEW BATCH!!!!!!!!! iWeightIndexBase = %d , blockDim.x = %d\n",iWeightIndexBase,blockDim.x);
		}*/

		// first, we copy d_WeightsThisTest to s_InputWeights (it is only a part of weights).
		// We don't have to use 'if(iWeightIndex < iNumOfWeights)', because memory for weights was already padded (now it's about 5% faster)
		for(int iCopiedWeightIndex = 0;iCopiedWeightIndex<p_iHowMuchMemoryForWeights;iCopiedWeightIndex += blockDim.x)
			s_InputWeights[iCopiedWeightIndex + threadIdx.x] = dp_pWeights[iCopiedWeightIndex + iWeightIndex];

		__syncthreads(); // We make sure that all data was written to shared memory

		int iFirstElementInThisBatch = iMoveWeightsForThisTest - iWeightIndexBase;
		int iLastElementInThisBatch = iFirstElementInThisBatch + p_iNumInputNeurons;

		// Not all threads are used in calulations
		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : iFirstElementInThisBatch %d , iLastElementInThisBatch %d , T1  = [%d] , T2 = [%d] , T3 = [%d]\n",blockIdx.x,threadIdx.x,iFirstElementInThisBatch,iLastElementInThisBatch,(threadIdx.x < p_iOutputNeuronCount),(iLastElementInThisBatch >= 0),(iFirstElementInThisBatch < 0 || iFirstElementInThisBatch < blockDim.x));
		if(threadIdx.x < p_iOutputNeuronCount && iLastElementInThisBatch >= 0 && (iFirstElementInThisBatch < 0 || iFirstElementInThisBatch < p_iHowMuchMemoryForWeights))
		{
			int iFirstWeightIndex = max(0,-iFirstElementInThisBatch);
			int iLastWeightIndex = min(p_iNumInputNeurons,p_iNumInputNeurons - (iLastElementInThisBatch - p_iHowMuchMemoryForWeights));
			for(int iWeightIndexToAdd = iFirstWeightIndex;iWeightIndexToAdd < iLastWeightIndex; ++iWeightIndexToAdd)
			{
				int iWeightIndexHere = iWeightIndexToAdd - iWeightIndexBase + iMoveWeightsForThisTest;
				//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d , iWeightIndexToAdd %d : d_LayerInputThisTest %f , d_WeightsThisTest %f , iWeightIndexHere %d, val[%d] %f , MULT %f\n",blockIdx.x,threadIdx.x,iWeightIndexToAdd,d_LayerInputThisTest[iWeightIndexToAdd],d_WeightsThisTest[iWeightIndexToAdd],iWeightIndexHere,iWeightIndexHere,s_InputWeights[iWeightIndexHere],d_LayerInputThisTest[iWeightIndexToAdd] * d_WeightsThisTest[iWeightIndexToAdd]);

				dResult += s_InputNeurons[iWeightIndexToAdd] * s_InputWeights[iWeightIndexHere];
			}
		}

		__syncthreads(); // We make sure that all data was read by all threads
	}

	if(threadIdx.x <= p_iOutputNeuronCount)
	{
		real_gpu dDerivativeOfLastOutput = 0.0f;

		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : dResult before output function %f\n",blockIdx.x,threadIdx.x,dResult);

		switch(p_eNeuronType)
		{		
			case Neuron::NT_LINEAR: 
				dDerivativeOfLastOutput = 1.0f;
				break;	// Do nothing
			case Neuron::NT_SIGMOID:
				real_gpu dExp = __expf(-dResult);
				dResult = 1.0f / (1.0f + dExp);
				dDerivativeOfLastOutput = dExp / __powf(1.0f + dExp,2);
				break;
		}
		
		if(threadIdx.x == p_iOutputNeuronCount)
			dResult = 1.0f; // bias

		//PRINT_DEBUG_INFO("XXXXXXXXXXXXXXXXXXXXXXXXXXXXGPU: Test %d , Neuron %d : %d\n",blockIdx.x,threadIdx.x,blockIdx.x*iNumOutputNeuronsAligned + threadIdx.x);
		*d_pLayerOutputThisTest = dResult;

		// We only need derivative of last output if we are in training!
		if(dp_pDerivativeOfLastOutput != NULL)
			*d_pDerivativeOfLastOutputThisTest = dDerivativeOfLastOutput;

		//PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d : first d_LayerInputThisTest %f , first d_WeightsThisTest %f , dResult %f , dDerivativeOfLastOutput %f\n",blockIdx.x,threadIdx.x,d_LayerInputThisTest[0],d_WeightsThisTest[0],dResult,dDerivativeOfLastOutput);
	}
}

extern "C" void executeLayerCUDA(const real_gpu *dp_pLayerInput,const real_gpu *dp_pWeights,real_gpu *dp_pLayerOutput,real_gpu *dp_pDerivativeOfLastOutput
								 ,int p_iTestCount,int p_iOutputNeuronCount,int p_iNumInputNeurons,Neuron::NeuronType p_eNeuronType,const int *p_pVecTestIndices)
{
	// blockDim.x should be a multiple of 16 (half warp). We will be able to retrieve global data using coalescing
	int iBlockDimUpdated = ALIGN_UP(p_iOutputNeuronCount+1,HALF_WARP);
	int iSharedMemorySize = p_iNumInputNeurons * sizeof(real_gpu); // memory for input


	int iNumOfWeights = p_iNumInputNeurons * p_iOutputNeuronCount;
	int iNumOfWeightsAligned = ALIGN_UP(iNumOfWeights,iBlockDimUpdated);
	int iHowMuchMemoryForWeights = (min(iNumOfWeightsAligned,512) / iBlockDimUpdated) * iBlockDimUpdated;

	iSharedMemorySize += iHowMuchMemoryForWeights * sizeof(real_gpu); // memory for weights

	// If p_pVecTestIndices!=NULL , then we use constant memory to set test indices for the kernel
	if(p_pVecTestIndices!=NULL)
	{
		cudaMemcpyToSymbol("iTestIndices",p_pVecTestIndices,p_iTestCount*sizeof(int),0);
	}

	int iNumInputNeuronsAligned = ALIGN_UP(p_iNumInputNeurons, HALF_WARP);

	executeLayerKernel <<<p_iTestCount,iBlockDimUpdated,iSharedMemorySize>>> (dp_pLayerInput,dp_pWeights,dp_pLayerOutput,dp_pDerivativeOfLastOutput,p_iNumInputNeurons
		,iNumInputNeuronsAligned,p_eNeuronType,p_iOutputNeuronCount,(p_pVecTestIndices!=NULL),iHowMuchMemoryForWeights);
}


__global__ void calculateErrorInLastLayerKernel(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors,int p_iSpaceBetweenTestsInOutput)
{
	int iElementIndexCorrectOutput = p_iSpaceBetweenTestsInOutput * iTestIndices[blockIdx.x] + threadIdx.x;
	int iElementIndex = p_iSpaceBetweenTestsInOutput * blockIdx.x + threadIdx.x;
	dp_pErrors[iElementIndex] = dp_pNetworkOutput[iElementIndex] - dp_pCorrectOutput[iElementIndexCorrectOutput];
	PRINT_DEBUG_INFO("GPU: Test in batch nr %d (test %d) , Output %d (iElementIndex %d) : Network = %f , Correct  = %f , Error = %f\n",blockIdx.x,iTestIndices[blockIdx.x],threadIdx.x
		,iElementIndex,dp_pNetworkOutput[iElementIndex],dp_pCorrectOutput[iElementIndex],dp_pErrors[iElementIndex]);
}

extern "C" void calculateErrorInLastLayerCUDA(const real_gpu *dp_pCorrectOutput,const real_gpu *dp_pNetworkOutput,real_gpu *dp_pErrors,int p_iOutputNeuronCount,int p_iNumTestsInBatch,int p_iSpaceBetweenTestsInOutput)
{
	calculateErrorInLastLayerKernel <<<p_iNumTestsInBatch,p_iOutputNeuronCount>>> (dp_pCorrectOutput,dp_pNetworkOutput,dp_pErrors,p_iSpaceBetweenTestsInOutput);
}


__global__ void calculateErrorInNotLastLayerKernel(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iNextLayerNeuronCount,int p_iNextLayerNeuronCountAligned,int p_iThisLayerNeuronCount)
{
	extern __shared__ real_gpu s_NextLayerErrorThisTest[];
	real_gpu dError = 0.0f;
	int iNextLayerWeightsForOneNeuron = blockDim.x + 1;
	const real_gpu *d_pNextLayerErrorThisTest = dp_pNextLayerError + p_iNextLayerNeuronCountAligned * blockIdx.x;

	// Copying error data from global to shared memory
	for(int iErrorIndex = threadIdx.x;iErrorIndex < p_iNextLayerNeuronCount; iErrorIndex += blockDim.x)
	{
		s_NextLayerErrorThisTest[iErrorIndex] = d_pNextLayerErrorThisTest[iErrorIndex];
	}

	if(threadIdx.x < p_iThisLayerNeuronCount)
	{
		for(int iWeightIndex = 0;iWeightIndex < p_iNextLayerNeuronCount; ++iWeightIndex)
		{
			PRINT_DEBUG_INFO("GPU: Test index %d , Neuron index %d , Weight index %d : dp_pNextLayerWeights [%d] = %f , dp_pNextLayerError[%d] = %f , MULT = %f\n"
				,blockIdx.x,threadIdx.x,iWeightIndex,iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x,dp_pNextLayerWeights[iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x],iWeightIndex
				,dp_pNextLayerError[iWeightIndex],dp_pNextLayerWeights[iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x] * dp_pNextLayerError[iWeightIndex]);
			dError += dp_pNextLayerWeights[iWeightIndex*iNextLayerWeightsForOneNeuron + threadIdx.x] * s_NextLayerErrorThisTest[iWeightIndex];
		}
		
		dp_pThisLayerError[blockDim.x*blockIdx.x + threadIdx.x] = dError;

		PRINT_DEBUG_INFO("GPU: Test index %d , Neuron index %d : Error %f\n",blockIdx.x,threadIdx.x,dError);
	}
}

extern "C" void calculateErrorInNotLastLayerCUDA(const real_gpu *dp_pNextLayerWeights,const real_gpu *dp_pNextLayerError,real_gpu *dp_pThisLayerError,int p_iThisLayerNeuronCount,int p_iNextLayerNeuronCount,int p_iNumTestsInBatch)
{
	int iElementsAllocatedForOneTestInNextLayerAligned = ALIGN_UP(p_iNextLayerNeuronCount+1,HALF_WARP);
	int iElementsAllocatedForOneTestInThisLayerAligned = ALIGN_UP(p_iThisLayerNeuronCount+1,HALF_WARP);
	int iSharedMemorySize = p_iNextLayerNeuronCount * sizeof(real_gpu); // memory for error

	calculateErrorInNotLastLayerKernel <<<p_iNumTestsInBatch,iElementsAllocatedForOneTestInThisLayerAligned,iSharedMemorySize>>> (dp_pNextLayerWeights,dp_pNextLayerError,dp_pThisLayerError,p_iNextLayerNeuronCount,iElementsAllocatedForOneTestInNextLayerAligned,p_iThisLayerNeuronCount);
}


__global__ void updateWeightsInTrainingKernel(const real_gpu *dp_pThisLayerError,const real_gpu *dp_pDerivativeOfLastOutput,const real_gpu *dp_pLayerBeforeOutputs,real_gpu p_dEta
											  ,real_gpu *dp_pThisLayerWeights,int p_iNumTestsInBatch,int iElementsAllocatedForOneTestInThisLayerAligned,int p_iElementsAllocatedForOneTestInLayerBeforeAligned,bool p_bLayerBeforeOutputsHaveSpecificIndexes)
{
	// We change: neuron blockIdx.x , weight threadIdx.x

	real_gpu dChange = 0.0f;
	for(unsigned uTestIndex = 0;uTestIndex < p_iNumTestsInBatch;++uTestIndex)
	{
		real_gpu dError = dp_pThisLayerError[iElementsAllocatedForOneTestInThisLayerAligned*uTestIndex + blockIdx.x];
		real_gpu dDerivativeOfLastOutput = dp_pDerivativeOfLastOutput[iElementsAllocatedForOneTestInThisLayerAligned*uTestIndex + blockIdx.x];

		int iTestIndexForOutputBefore = ( p_bLayerBeforeOutputsHaveSpecificIndexes ? iTestIndices[uTestIndex] : uTestIndex );
		real_gpu dLayerBeforeOutput = dp_pLayerBeforeOutputs[p_iElementsAllocatedForOneTestInLayerBeforeAligned*iTestIndexForOutputBefore + threadIdx.x];

		real_gpu dChangeThisTest = dError * dDerivativeOfLastOutput * dLayerBeforeOutput * p_dEta;
		PRINT_DEBUG_INFO("GPU: Test %d , Neuron %d , Weight %d : dError %f , dDerivativeOfLastOutput %f , dLayerBeforeOutput %f , dChangeThisTest %f\n",uTestIndex,blockIdx.x,threadIdx.x,dError,dDerivativeOfLastOutput,dLayerBeforeOutput,dChangeThisTest);
		dChange += dChangeThisTest;
	}

	//int iTestIndexForWeights = ( p_bLayerBeforeOutputsHaveSpecificIndexes ? iTestIndices[blockIdx.x] : blockIdx.x );
	int iWeightIndex = blockDim.x*blockIdx.x + threadIdx.x;
	PRINT_DEBUG_INFO("GPU: Neuron %d , Weight %d (index in array %d) : Old weight %f , Change %f , New weight %f\n",blockIdx.x,threadIdx.x,iWeightIndex,dp_pThisLayerWeights[iWeightIndex],dChange,dp_pThisLayerWeights[iWeightIndex] - dChange);
	dp_pThisLayerWeights[iWeightIndex] = dp_pThisLayerWeights[iWeightIndex] - dChange;

	/*real_gpu *d_pThisNeuronWeights = dp_pThisLayerWeights + threadIdx.x * (p_iNumOutputsLayerBefore+1);
	real_gpu dErrorMultDerivativeMultEta = dp_pThisLayerError[threadIdx.x] * dp_pDerivativeOfLastOutput[threadIdx.x] * p_dEta;

	PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d : First weight: %f , dErrorMultDerivativeMultEta: %f , p_iNumOutputsLayerBefore: %d\n",threadIdx.x,d_pThisNeuronWeights[0],dErrorMultDerivativeMultEta,p_iNumOutputsLayerBefore);
	
	for(unsigned uWeightIndex = 0;uWeightIndex < p_iNumOutputsLayerBefore;++uWeightIndex)
	{
		real_gpu dChange = dErrorMultDerivativeMultEta * dp_pLayerBeforeOutputs[uWeightIndex];
		real_gpu dCurrentValue = d_pThisNeuronWeights[uWeightIndex];
		real_gpu dChangedValue = dCurrentValue - dChange;
		PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d , uWeightIndex %d: dCurrentValue = %f , dChange %f , dChangedValue %f\n",threadIdx.x,uWeightIndex,dCurrentValue,dChange,dChangedValue);
		d_pThisNeuronWeights[uWeightIndex] = dChangedValue;
	}
	
	PRINT_DEBUG_INFO("GPU: Test 0 , Neuron %d , Bias : dChange %f\n",threadIdx.x,dErrorMultDerivativeMultEta);
	d_pThisNeuronWeights[p_iNumOutputsLayerBefore] = d_pThisNeuronWeights[p_iNumOutputsLayerBefore] - dErrorMultDerivativeMultEta;*/
}

extern "C" void updateWeightsInTrainingCUDA(const real_gpu *dp_pThisLayerError,const real_gpu *dp_pDerivativeOfLastOutput,const real_gpu *dp_pLayerBeforeOutputs,real_gpu p_dEta,int p_iThisLayerNeuronCount
											,int p_iNumOutputsLayerBefore,real_gpu *dp_pThisLayerWeights,int p_iNumTestsInBatch,bool p_bLayerBeforeOutputsHaveSpecificIndexes)
{
	int iElementsAllocatedForOneTestInThisLayerAligned = ALIGN_UP(p_iThisLayerNeuronCount+1,HALF_WARP);
	int iElementsAllocatedForOneTestInLayerBeforeAligned = ALIGN_UP(p_iNumOutputsLayerBefore+1,HALF_WARP);

	updateWeightsInTrainingKernel <<<p_iThisLayerNeuronCount,p_iNumOutputsLayerBefore+1>>> (dp_pThisLayerError,dp_pDerivativeOfLastOutput,dp_pLayerBeforeOutputs,p_dEta
		,dp_pThisLayerWeights,p_iNumTestsInBatch,iElementsAllocatedForOneTestInThisLayerAligned,iElementsAllocatedForOneTestInLayerBeforeAligned,p_bLayerBeforeOutputsHaveSpecificIndexes);
}
