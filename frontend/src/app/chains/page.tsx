'use client';

import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';

interface Contract {
  address: string;
  name: string;
  network: string;
  deployed_at: string;
  verified: boolean;
}

interface ShardStatus {
  id: number;
  name: string;
  type: string;
  status: string;
  tps: number;
  load: number;
  validators: number;
  uptime: number;
}

const API_BASE_URL = 'http://localhost:8000/api/v1';

const CONTRACT_TEMPLATES = {
  InferenceRegistry: `pragma solidity ^0.8.19;

contract InferenceRegistry {
    struct Inference {
        address provider;
        string modelHash;
        string resultHash;
        uint256 timestamp;
    }
    
    mapping(uint256 => Inference) public inferences;
    uint256 public inferenceCount;
    
    event InferenceRegistered(uint256 indexed id, address provider);
    
    function registerInference(string memory modelHash, string memory resultHash) public {
        inferences[inferenceCount] = Inference(msg.sender, modelHash, resultHash, block.timestamp);
        emit InferenceRegistered(inferenceCount, msg.sender);
        inferenceCount++;
    }
}`,
  ShardAwareInferenceRegistry: `pragma solidity ^0.8.19;

contract ShardAwareInferenceRegistry {
    uint256 public shardId;
    
    struct Inference {
        address provider;
        string modelHash;
        uint256 shardId;
        uint256 timestamp;
    }
    
    mapping(uint256 => Inference) public inferences;
    
    constructor(uint256 _shardId) {
        shardId = _shardId;
    }
    
    function registerInference(string memory modelHash) public {
        inferences[block.timestamp] = Inference(msg.sender, modelHash, shardId, block.timestamp);
    }
}`,
  AEGISToken: `pragma solidity ^0.8.19;

contract AEGISToken {
    string public name = "AEGIS Token";
    string public symbol = "AEGIS";
    uint256 public totalSupply;
    
    mapping(address => uint256) public balanceOf;
    
    event Transfer(address indexed from, address indexed to, uint256 value);
    
    constructor(uint256 _initialSupply) {
        totalSupply = _initialSupply;
        balanceOf[msg.sender] = _initialSupply;
    }
    
    function transfer(address _to, uint256 _value) public returns (bool) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }
}`
};

const NETWORKS = [
  { id: 'polygon-mumbai', name: 'Polygon Mumbai (Testnet)' },
  { id: 'polygon', name: 'Polygon PoS (Mainnet)' },
  { id: 'clover-testnet', name: 'Clover Testnet' },
  { id: 'clover', name: 'Clover Mainnet' },
  { id: 'aegis-shard-0', name: 'AEGIS Shard 0 (Inference)' },
  { id: 'aegis-shard-1', name: 'AEGIS Shard 1 (Storage)' },
  { id: 'aegis-shard-2', name: 'AEGIS Shard 2 (Governance)' }
];

export default function ChainsPage() {
  const [selectedContract, setSelectedContract] = useState('InferenceRegistry');
  const [selectedNetwork, setSelectedNetwork] = useState('polygon-mumbai');
  const [sourceCode, setSourceCode] = useState(CONTRACT_TEMPLATES.InferenceRegistry);
  const [deployedContracts, setDeployedContracts] = useState<Contract[]>([]);
  const [shards, setShards] = useState<ShardStatus[]>([]);
  const [isCompiling, setIsCompiling] = useState(false);
  const [isDeploying, setIsDeploying] = useState(false);
  const [message, setMessage] = useState('');

  const handleContractChange = (contractName: string) => {
    setSelectedContract(contractName);
    setSourceCode(CONTRACT_TEMPLATES[contractName as keyof typeof CONTRACT_TEMPLATES]);
  };

  const handleCompile = async () => {
    setIsCompiling(true);
    setMessage('');
    try {
      const response = await fetch(`${API_BASE_URL}/blockchain/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_code: sourceCode,
          compiler_version: '0.8.19',
          optimization: true
        })
      });
      const data = await response.json();
      if (data.success) {
        setMessage('✅ Contract compiled successfully!');
      } else {
        setMessage('❌ Compilation failed');
      }
    } catch (error) {
      setMessage('❌ Error: ' + (error as Error).message);
    } finally {
      setIsCompiling(false);
    }
  };

  const handleDeploy = async () => {
    setIsDeploying(true);
    setMessage('');
    try {
      const response = await fetch(`${API_BASE_URL}/blockchain/deploy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contract_name: selectedContract,
          network: selectedNetwork,
          constructor_args: [],
          source_code: sourceCode
        })
      });
      const data = await response.json();
      if (data.success) {
        setMessage(`✅ Contract deployed at: ${data.contract_address}`);
        loadContracts();
      } else {
        setMessage('❌ Deployment failed');
      }
    } catch (error) {
      setMessage('❌ Error: ' + (error as Error).message);
    } finally {
      setIsDeploying(false);
    }
  };

  const loadContracts = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/blockchain/contracts`);
      const data = await response.json();
      if (data.success) {
        setDeployedContracts(data.contracts);
      }
    } catch (error) {
      console.error('Error loading contracts:', error);
    }
  };

  const loadShardStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/blockchain/shard-status`);
      const data = await response.json();
      if (data.success) {
        setShards(data.shards);
      }
    } catch (error) {
      console.error('Error loading shard status:', error);
    }
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold">Blockchain & Smart Contracts</h1>
        <div className="flex gap-2">
          <Button onClick={loadContracts} variant="outline">
            Refresh Contracts
          </Button>
          <Button onClick={loadShardStatus} variant="outline">
            Load Shard Status
          </Button>
        </div>
      </div>

      {/* Contract Selection */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Deploy Smart Contract</h2>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-2">Contract Template</label>
            <select
              value={selectedContract}
              onChange={(e) => handleContractChange(e.target.value)}
              className="w-full p-2 border rounded"
            >
              <option value="InferenceRegistry">Inference Registry</option>
              <option value="ShardAwareInferenceRegistry">Shard-Aware Inference Registry</option>
              <option value="AEGISToken">AEGIS Token</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">Network</label>
            <select
              value={selectedNetwork}
              onChange={(e) => setSelectedNetwork(e.target.value)}
              className="w-full p-2 border rounded"
            >
              {NETWORKS.map(network => (
                <option key={network.id} value={network.id}>{network.name}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium mb-2">Source Code</label>
          <textarea
            value={sourceCode}
            onChange={(e) => setSourceCode(e.target.value)}
            className="w-full h-64 p-3 font-mono text-sm border rounded"
          />
        </div>

        <div className="flex gap-3">
          <Button onClick={handleCompile} disabled={isCompiling}>
            {isCompiling ? 'Compiling...' : 'Compile Contract'}
          </Button>
          <Button onClick={handleDeploy} disabled={isDeploying}>
            {isDeploying ? 'Deploying...' : 'Deploy Contract'}
          </Button>
        </div>

        {message && (
          <div className="mt-4 p-3 bg-gray-100 rounded">
            {message}
          </div>
        )}
      </Card>

      {/* Deployed Contracts */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Deployed Contracts ({deployedContracts.length})</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {deployedContracts.map((contract) => (
            <Card key={contract.address} className="p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="font-semibold">{contract.name}</h3>
                {contract.verified && <Badge>Verified</Badge>}
              </div>
              <p className="text-xs text-gray-600 mb-1">Network: {contract.network}</p>
              <p className="text-xs font-mono bg-gray-100 p-2 rounded mb-2 truncate">
                {contract.address}
              </p>
              <p className="text-xs text-gray-500">
                {new Date(contract.deployed_at).toLocaleString()}
              </p>
            </Card>
          ))}
        </div>
      </Card>

      {/* Shard Status */}
      {shards.length > 0 && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">AEGIS Shard Status</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {shards.map((shard) => (
              <Card key={shard.id} className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold">{shard.name}</h3>
                  <Badge variant={shard.status === 'active' ? 'default' : 'secondary'}>
                    {shard.status}
                  </Badge>
                </div>
                <div className="space-y-1 text-sm">
                  <p>Type: <span className="font-medium">{shard.type}</span></p>
                  <p>TPS: <span className="font-medium">{shard.tps.toLocaleString()}</span></p>
                  <p>Load: <span className="font-medium">{shard.load}%</span></p>
                  <p>Validators: <span className="font-medium">{shard.validators}</span></p>
                  <p>Uptime: <span className="font-medium">{shard.uptime}%</span></p>
                </div>
              </Card>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
